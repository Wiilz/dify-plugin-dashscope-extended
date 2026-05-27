from typing import Mapping

from dashscope.common.error import (
    AuthenticationError,
    InvalidParameter,
    RequestFailure,
    ServiceUnavailableError,
    UnsupportedHTTPMethod,
    UnsupportedModel,
)

from dify_plugin.errors.model import (
    InvokeAuthorizationError,
    InvokeBadRequestError,
    InvokeConnectionError,
    InvokeError,
    InvokeRateLimitError,
    InvokeServerUnavailableError,
)

DEFAULT_HTTP_BASE_ADDRESS = "https://dashscope.aliyuncs.com/api/v1"
INTL_HTTP_BASE_ADDRESS = "https://dashscope-intl.aliyuncs.com/api/v1"


def get_http_base_address(credentials: Mapping[str, str]) -> str:
    if credentials.get("use_international_endpoint", "false") == "true":
        return INTL_HTTP_BASE_ADDRESS
    return DEFAULT_HTTP_BASE_ADDRESS


class _CommonDashScope:
    @staticmethod
    def _to_credential_kwargs(credentials: dict) -> dict:
        return {"api_key": credentials["dashscope_api_key"]}

    @property
    def _invoke_error_mapping(self) -> dict[type[InvokeError], list[type[Exception]]]:
        return {
            InvokeConnectionError: [RequestFailure],
            InvokeServerUnavailableError: [ServiceUnavailableError],
            InvokeRateLimitError: [],
            InvokeAuthorizationError: [AuthenticationError],
            InvokeBadRequestError: [
                InvalidParameter,
                UnsupportedModel,
                UnsupportedHTTPMethod,
            ],
        }
