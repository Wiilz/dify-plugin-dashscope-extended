import logging
from collections.abc import Mapping

from dashscope import Generation

from dify_plugin import ModelProvider
from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import UserPromptMessage
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class DashscopeExtendedModelProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: Mapping) -> None:
        try:
            response = Generation.call(
                api_key=credentials["dashscope_api_key"],
                model="qwen-turbo",
                messages=[{"role": "user", "content": "ping"}],
                stream=False,
                max_tokens=1,
            )
            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f"Credential validation failed: {response.code} - {response.message}"
                )
        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            logger.exception("dashscope-extended provider credentials validate failed")
            raise CredentialsValidateFailedError(str(ex))
