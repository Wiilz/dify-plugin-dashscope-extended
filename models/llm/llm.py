import base64
import json
import logging
import os
import sys
import tempfile
import uuid
from collections.abc import Generator
from http import HTTPStatus
from pathlib import Path
from typing import Optional, Union, cast

import requests
from dashscope import Generation, MultiModalConversation, get_tokenizer
from dashscope.api_entities.dashscope_response import GenerationResponse
from dashscope.common.error import (
    AuthenticationError,
    InvalidParameter,
    RequestFailure,
    ServiceUnavailableError,
    UnsupportedHTTPMethod,
    UnsupportedModel,
)
from dify_plugin.entities.model import (
    AIModelEntity,
    FetchFrom,
    I18nObject,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
)
from dify_plugin.entities.model.llm import (
    LLMMode,
    LLMResult,
    LLMResultChunk,
    LLMResultChunkDelta,
)
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    AudioPromptMessageContent,
    DocumentPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageRole,
    PromptMessageTool,
    SystemPromptMessage,
    TextPromptMessageContent,
    ToolPromptMessage,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from dify_plugin.errors.model import (
    CredentialsValidateFailedError,
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel
from openai import OpenAI
from models._common import get_http_base_address
from dify_plugin.config.logger_format import plugin_logger_handler


class _Utf8Formatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "event": "log",
            "data": {
                "level": record.levelname,
                "message": record.getMessage(),
                "timestamp": record.created,
            },
        }, ensure_ascii=False)


_handler = logging.StreamHandler(sys.stdout)
_handler.setLevel(logging.INFO)
_handler.setFormatter(_Utf8Formatter())

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(_handler)


class DashscopeExtendedLargeLanguageModel(LargeLanguageModel):
    tokenizers = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._temp_files = []

    def _invoke(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: dict,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[list[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        return self._generate(
            model, credentials, prompt_messages, model_parameters,
            tools, stop, stream, user,
        )

    def get_num_tokens(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        if self.get_customizable_model_schema(model, credentials) is not None:
            return 0
        if model in self.tokenizers:
            tokenizer = self.tokenizers[model]
        else:
            try:
                tokenizer = get_tokenizer(model)
            except Exception:
                return 0
            self.tokenizers[model] = tokenizer
        tokens = tokenizer.encode(self._convert_messages_to_prompt(prompt_messages))
        return len(tokens)

    def validate_credentials(self, model: str, credentials: dict) -> None:
        try:
            self._generate(
                model=model,
                credentials=credentials,
                prompt_messages=[UserPromptMessage(content="ping")],
                model_parameters={"temperature": 0.5},
                stream=False,
            )
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex))

    def _generate(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: dict,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[list[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        credentials_kwargs = {"api_key": credentials["dashscope_api_key"]}
        model_schema = self.get_model_schema(model, credentials)

        logger.info(
            f"[dashscope-extended] _generate called: model={model}, "
            f"model_parameters={list(model_parameters.keys())}, "
            f"tools={len(tools) if tools else 0}, stream={stream}"
        )

        extra_model_kwargs = {}
        if tools:
            extra_model_kwargs["tools"] = self._convert_tools(tools)
        if stop:
            extra_model_kwargs["stop"] = stop

        # response_format: "json_object" -> {"type": "json_object"}
        response_format = model_parameters.get("response_format")
        if response_format:
            model_parameters["response_format"] = {"type": response_format}

        # search_options: 解析 JSON 字符串为 dict（兼容 Dify 对 text 类型参数自动加 _json 后缀的行为）
        search_options_str = (
                model_parameters.pop("search_options", None)
                or model_parameters.pop("search_options_json", None)
        )
        if search_options_str:
            try:
                search_options = json.loads(search_options_str)
                extra_model_kwargs["search_options"] = search_options
            except (json.JSONDecodeError, TypeError):
                raise InvokeBadRequestError(
                    "search_options 不是有效的 JSON 格式，请检查输入。"
                )

        # parallel_tool_calls: 仅在有工具时传递
        parallel_tool_calls = model_parameters.pop("parallel_tool_calls", None)
        if parallel_tool_calls is not None and tools:
            extra_model_kwargs["parallel_tool_calls"] = parallel_tool_calls

        # enable_thinking: 通用支持，直接透传
        enable_thinking = model_parameters.pop("enable_thinking", None)
        if enable_thinking is not None:
            extra_model_kwargs["enable_thinking"] = enable_thinking

        # reasoning_effort 与 thinking_budget 互斥（qwen3.8-max 等模型同时设置会报错）。
        # reasoning_effort 带默认值会被每次下发，而 thinking_budget 需用户显式填写，
        # 因此两者并存时以用户显式填写的 thinking_budget 为准。
        if model_parameters.get("reasoning_effort") and model_parameters.get("thinking_budget"):
            model_parameters.pop("reasoning_effort")
            logger.info(
                "[dashscope-extended] reasoning_effort 与 thinking_budget 互斥，"
                "已保留 thinking_budget 并忽略 reasoning_effort。"
            )

        params = {
            "model": model,
            **model_parameters,
            **credentials_kwargs,
            **extra_model_kwargs,
        }

        base_address = get_http_base_address(credentials)

        # 视觉模型路由：基于 YAML features 声明
        if ModelFeature.VISION in (model_schema.features or []):
            params["messages"] = self._convert_prompt_messages_to_tongyi_messages(
                credentials, prompt_messages, rich_content=True
            )
            # 打印完整请求参数（隐藏 api_key）
            log_params = {k: v for k, v in params.items() if k != "api_key"}
            logger.info(
                f"[dashscope-extended] MultiModalConversation 请求参数: {json.dumps(log_params, ensure_ascii=False, default=str)}")
            response = MultiModalConversation.call(
                **params,
                stream=True,
                incremental_output=True,
                base_address=base_address,
            )
        else:
            params["messages"] = self._convert_prompt_messages_to_tongyi_messages(
                credentials, prompt_messages
            )
            # 打印完整请求参数（隐藏 api_key）
            log_params = {k: v for k, v in params.items() if k != "api_key"}
            logger.info(
                f"[dashscope-extended] Generation 请求参数: {json.dumps(log_params, ensure_ascii=False, default=str)}")
            response = Generation.call(
                **params,
                result_format="message",
                stream=True,
                incremental_output=True,
                base_address=base_address,
            )

        return self._handle_generate_stream_response(
            model, credentials, response, prompt_messages,
        )

    def _handle_generate_response(
            self,
            model: str,
            credentials: dict,
            response: GenerationResponse,
            prompt_messages: list[PromptMessage],
    ) -> LLMResult:
        try:
            if response.status_code not in {200, HTTPStatus.OK}:
                self._handle_error_response(response.status_code, response.message, model)

            resp_content = response.output.choices[0].message.content
            if isinstance(resp_content, list):
                resp_content = resp_content[0]["text"]

            assistant_prompt_message = AssistantPromptMessage(
                content=resp_content,
                tool_calls=response.output.choices[0].message.get("tool_calls", []),
            )
            usage = self._calc_response_usage(
                model, credentials,
                response.usage.input_tokens,
                response.usage.output_tokens,
            )
            return LLMResult(
                model=model,
                message=assistant_prompt_message,
                prompt_messages=prompt_messages,
                usage=usage,
            )
        finally:
            self._cleanup_temp_files()

    def _handle_tool_call_stream(self, response, tool_calls, incremental_output):
        tool_calls_stream = response.output.choices[0].message["tool_calls"]
        for tool_call_stream in tool_calls_stream:
            idx = tool_call_stream.get("index")
            if idx >= len(tool_calls):
                tool_calls.append(tool_call_stream)
            else:
                if tool_call_stream.get("function"):
                    func_name = tool_call_stream.get("function").get("name")
                    tool_call_obj = tool_calls[idx]
                    if func_name:
                        if incremental_output:
                            tool_call_obj["function"]["name"] += func_name
                        else:
                            tool_call_obj["function"]["name"] = func_name
                    args = tool_call_stream.get("function").get("arguments")
                    if args:
                        if incremental_output:
                            tool_call_obj["function"]["arguments"] += args
                        else:
                            tool_call_obj["function"]["arguments"] = args

    def _handle_generate_stream_response(
            self,
            model: str,
            credentials: dict,
            responses: Generator[GenerationResponse, None, None],
            prompt_messages: list[PromptMessage],
    ) -> Generator:
        is_reasoning = False
        tool_calls = []
        try:
            for index, response in enumerate(responses):
                if response.status_code not in {200, HTTPStatus.OK}:
                    request_id = getattr(response, "request_id", None)
                    self._handle_error_response(
                        response.status_code, response.message, model, request_id
                    )

                resp_finish_reason = response.output.choices[0].finish_reason
                if resp_finish_reason is not None and resp_finish_reason != "null":
                    resp_content = response.output.choices[0].message.content
                    assistant_prompt_message = AssistantPromptMessage(content="")

                    if "tool_calls" in response.output.choices[0].message:
                        self._handle_tool_call_stream(response, tool_calls, True)
                    elif resp_content:
                        if isinstance(resp_content, list):
                            resp_content = resp_content[0]["text"]
                        assistant_prompt_message.content = resp_content
                    elif is_reasoning:
                        assistant_prompt_message.content = "\n</think>"

                    if tool_calls:
                        message_tool_calls = []
                        for tool_call_obj in tool_calls:
                            message_tool_call = AssistantPromptMessage.ToolCall(
                                id=tool_call_obj["function"]["name"],
                                type="function",
                                function=AssistantPromptMessage.ToolCall.ToolCallFunction(
                                    name=tool_call_obj["function"]["name"],
                                    arguments=tool_call_obj["function"]["arguments"],
                                ),
                            )
                            message_tool_calls.append(message_tool_call)
                        assistant_prompt_message.tool_calls = message_tool_calls

                    usage = response.usage
                    usage = self._calc_response_usage(
                        model, credentials, usage.input_tokens, usage.output_tokens,
                    )
                    yield LLMResultChunk(
                        model=model,
                        prompt_messages=prompt_messages,
                        delta=LLMResultChunkDelta(
                            index=index,
                            message=assistant_prompt_message,
                            finish_reason=resp_finish_reason,
                            usage=usage,
                        ),
                    )
                else:
                    message = response.output.choices[0].message
                    resp_content, is_reasoning = self._wrap_thinking_by_reasoning_content(
                        message, is_reasoning
                    )

                    content_to_yield = []
                    if resp_content:
                        content_to_yield.append(resp_content)

                    if "tool_calls" in message:
                        if is_reasoning:
                            content_to_yield.append("\n</think>")
                            is_reasoning = False
                        self._handle_tool_call_stream(response, tool_calls, True)

                    if content_to_yield:
                        assistant_prompt_message = AssistantPromptMessage(
                            content="".join(content_to_yield)
                        )
                        yield LLMResultChunk(
                            model=model,
                            prompt_messages=prompt_messages,
                            delta=LLMResultChunkDelta(
                                index=index,
                                message=assistant_prompt_message,
                            ),
                        )
        finally:
            self._cleanup_temp_files()

    def _convert_one_message_to_text(self, message: PromptMessage) -> str:
        human_prompt = "\n\nHuman:"
        ai_prompt = "\n\nAssistant:"
        content = message.content
        if isinstance(message, UserPromptMessage):
            if isinstance(content, str):
                message_text = f"{human_prompt} {content}"
            elif isinstance(content, list):
                message_text = ""
                for sub_message in content:
                    if sub_message.type == PromptMessageContentType.TEXT:
                        message_text = f"{human_prompt} {sub_message.data}"
                        break
            else:
                raise TypeError(f"Unexpected content type: {type(content)}")
        elif isinstance(message, AssistantPromptMessage):
            message_text = f"{ai_prompt} {content}"
        elif isinstance(message, SystemPromptMessage | ToolPromptMessage):
            message_text = content
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _convert_messages_to_prompt(self, messages: list[PromptMessage]) -> str:
        messages = messages.copy()
        text = "".join(
            self._convert_one_message_to_text(message) for message in messages
        )
        return text.rstrip()

    def _convert_prompt_messages_to_tongyi_messages(
            self,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            rich_content: bool = False,
    ) -> list[dict]:
        tongyi_messages = []
        for prompt_message in prompt_messages:
            if isinstance(prompt_message, SystemPromptMessage):
                tongyi_messages.append({
                    "role": "system",
                    "content": (
                        prompt_message.content
                        if not rich_content
                        else [{"text": prompt_message.content}]
                    ),
                })
            elif isinstance(prompt_message, UserPromptMessage):
                if isinstance(prompt_message.content, str):
                    tongyi_messages.append({
                        "role": "user",
                        "content": (
                            prompt_message.content
                            if not rich_content
                            else [{"text": prompt_message.content}]
                        ),
                    })
                else:
                    user_messages = []
                    file_id_list = []
                    for message_content in prompt_message.content:
                        if message_content.type == PromptMessageContentType.TEXT:
                            message_content = cast(TextPromptMessageContent, message_content)
                            user_messages.append({"text": message_content.data})
                        elif message_content.type == PromptMessageContentType.IMAGE:
                            message_content = cast(ImagePromptMessageContent, message_content)
                            image_url = message_content.data
                            if message_content.data.startswith("data:"):
                                image_url = self._save_base64_to_file(message_content.data)
                            user_messages.append({"image": image_url})
                        elif message_content.type == PromptMessageContentType.VIDEO:
                            message_content = cast(VideoPromptMessageContent, message_content)
                            video_url = message_content.data
                            if message_content.data.startswith("data:"):
                                video_url = self._save_base64_to_file(message_content.data)
                            user_messages.append({"video": video_url})
                        elif message_content.type == PromptMessageContentType.AUDIO:
                            message_content = cast(AudioPromptMessageContent, message_content)
                            audio_data = message_content.data
                            if not audio_data:
                                raise ValueError("Audio content cannot be empty.")
                            if audio_data.startswith("data:"):
                                audio_data = self._save_base64_to_file(audio_data)
                            user_messages.append({"audio": audio_data})
                        elif message_content.type == PromptMessageContentType.DOCUMENT:
                            message_content = cast(DocumentPromptMessageContent, message_content)
                            file_id = self._upload_file_to_dashscope(credentials, message_content)
                            file_id_list.append(f"fileid://{file_id}")
                    if file_id_list:
                        tongyi_messages.append({
                            "role": "system",
                            "content": ",".join(file_id_list),
                        })
                    user_messages = sorted(user_messages, key=lambda x: "text" in x)
                    tongyi_messages.append({"role": "user", "content": user_messages})
            elif isinstance(prompt_message, AssistantPromptMessage):
                content = prompt_message.content or " "
                message = {
                    "role": "assistant",
                    "content": content if not rich_content else [{"text": content}],
                }
                if prompt_message.tool_calls:
                    message["tool_calls"] = [
                        tool_call.model_dump() for tool_call in prompt_message.tool_calls
                    ]
                tongyi_messages.append(message)
            elif isinstance(prompt_message, ToolPromptMessage):
                tongyi_messages.append({
                    "role": "tool",
                    "content": prompt_message.content,
                    "name": prompt_message.tool_call_id,
                })
            else:
                raise ValueError(f"Got unknown type {prompt_message}")
        return tongyi_messages

    def _save_base64_to_file(self, base64_data: str) -> str:
        (mime_type, encoded_string) = (
            base64_data.split(",")[0].split(";")[0].split(":")[1],
            base64_data.split(",")[1],
        )
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, f"{uuid.uuid4()}.{mime_type.split('/')[1]}")
        Path(file_path).write_bytes(base64.b64decode(encoded_string))
        self._temp_files.append(file_path)
        return f"file://{file_path}"

    def _cleanup_temp_files(self):
        for file_path in self._temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        self._temp_files.clear()

    def _upload_file_to_dashscope(
            self, credentials: dict, message_content: DocumentPromptMessageContent
    ) -> str:
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if credentials.get("use_international_endpoint", "false") == "true":
            base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        client = OpenAI(
            api_key=credentials["dashscope_api_key"],
            base_url=base_url,
        )
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file_path = temp_file.name
                if message_content.base64_data:
                    temp_file.write(base64.b64decode(message_content.base64_data))
                else:
                    try:
                        response = requests.get(message_content.url, timeout=60)
                        response.raise_for_status()
                        temp_file.write(response.content)
                    except Exception as ex:
                        raise ValueError(
                            f"Failed to fetch data from url {message_content.url}, {ex}"
                        ) from ex
                temp_file.flush()
            with open(temp_file_path, "rb") as f:
                response = client.files.create(file=f, purpose="file-extract")
            return response.id
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temporary file {temp_file_path}: {e}")

    def _convert_tools(self, tools: list[PromptMessageTool]) -> list[dict]:
        tool_definitions = []
        for tool in tools:
            properties = tool.parameters["properties"]
            required_properties = tool.parameters["required"]
            properties_definitions = {}
            for p_key, p_val in properties.items():
                desc = p_val.get("description") or ""
                if "enum" in p_val:
                    desc += f"; Only accepts one of: [{', '.join(p_val['enum'])}]"
                properties_definitions[p_key] = {
                    "description": desc,
                    "type": p_val["type"],
                }
            tool_definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": properties_definitions,
                    "required": required_properties,
                },
            })
        return tool_definitions

    def _wrap_thinking_by_reasoning_content(
            self, delta: dict, is_reasoning: bool
    ) -> tuple[str, bool]:
        content = delta.get("content") or ""
        if isinstance(content, list) and content:
            content = content[0].get("text") if isinstance(content[0], dict) else ""
        else:
            content = str(content)

        reasoning_content = delta.get("reasoning_content")
        try:
            if reasoning_content:
                try:
                    if isinstance(reasoning_content, list):
                        reasoning_content = "\n".join(map(str, reasoning_content))
                    elif not isinstance(reasoning_content, str):
                        reasoning_content = str(reasoning_content)

                    if not is_reasoning:
                        content = "<think>\n" + reasoning_content
                        is_reasoning = True
                    else:
                        content = reasoning_content
                except Exception as ex:
                    raise ValueError(f"[wrap_thinking-1] {ex}") from ex
            elif is_reasoning and content:
                content = "\n</think>" + content
                is_reasoning = False
        except Exception as ex:
            raise ValueError(f"[wrap_thinking-2] {ex}") from ex
        return content, is_reasoning

    def _handle_error_response(
            self, status_code: int, message: str, model: str = None, request_id: str = None
    ) -> None:
        error_msg = f"Failed to invoke model {model}, status_code: {status_code}, message: {message}" if model else message
        if request_id:
            error_msg += f", request_id: {request_id}"

        if status_code == 400:
            raise InvokeBadRequestError(error_msg)
        elif status_code in {401, 403}:
            raise InvokeAuthorizationError(error_msg)
        elif status_code == 422:
            raise InvokeBadRequestError(error_msg)
        elif status_code == 429:
            raise InvokeRateLimitError(error_msg)
        elif status_code >= 500:
            raise InvokeServerUnavailableError(error_msg)
        elif 400 <= status_code < 500:
            raise InvokeBadRequestError(error_msg)
        else:
            raise InvokeServerUnavailableError(error_msg)

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [RequestFailure],
            InvokeServerUnavailableError: [ServiceUnavailableError],
            InvokeRateLimitError: [],
            InvokeAuthorizationError: [AuthenticationError],
            InvokeBadRequestError: [InvalidParameter, UnsupportedModel, UnsupportedHTTPMethod],
        }

    def get_customizable_model_schema(
            self, model: str, credentials: dict
    ) -> Optional[AIModelEntity]:
        return AIModelEntity(
            model=model,
            label=I18nObject(en_US=model, zh_Hans=model),
            model_type=ModelType.LLM,
            features=(
                [
                    ModelFeature.TOOL_CALL,
                    ModelFeature.MULTI_TOOL_CALL,
                    ModelFeature.STREAM_TOOL_CALL,
                ]
                if credentials.get("function_calling_type") == "tool_call"
                else []
            ),
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(credentials.get("context_size", 8000)),
                ModelPropertyKey.MODE: LLMMode.CHAT.value,
            },
            parameter_rules=[
                ParameterRule(
                    name="temperature",
                    use_template="temperature",
                    label=I18nObject(en_US="Temperature", zh_Hans="温度"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="max_tokens",
                    use_template="max_tokens",
                    default=512,
                    min=1,
                    max=int(credentials.get("max_tokens", 1024)),
                    label=I18nObject(en_US="Max Tokens", zh_Hans="最大标记"),
                    type=ParameterType.INT,
                ),
                ParameterRule(
                    name="top_p",
                    use_template="top_p",
                    label=I18nObject(en_US="Top P", zh_Hans="Top P"),
                    type=ParameterType.FLOAT,
                ),
                ParameterRule(
                    name="enable_thinking",
                    label=I18nObject(en_US="Deep Thinking", zh_Hans="深度思考"),
                    type=ParameterType.BOOLEAN,
                    default=False,
                ),
                ParameterRule(
                    name="enable_search",
                    label=I18nObject(en_US="Web Search", zh_Hans="联网搜索"),
                    type=ParameterType.BOOLEAN,
                    default=False,
                ),
                ParameterRule(
                    name="search_options",
                    label=I18nObject(en_US="Search Options (JSON)", zh_Hans="搜索选项（JSON）"),
                    type=ParameterType.STRING,
                ),
                ParameterRule(
                    name="parallel_tool_calls",
                    label=I18nObject(en_US="Parallel Tool Calls", zh_Hans="并行工具调用"),
                    type=ParameterType.BOOLEAN,
                    default=True,
                ),
            ],
        )
