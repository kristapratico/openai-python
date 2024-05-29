from __future__ import annotations

from typing import Union
from typing_extensions import Literal

import pytest

from openai._models import FinalRequestOptions
from openai.lib.azure import AzureOpenAI, AsyncAzureOpenAI

Client = Union[AzureOpenAI, AsyncAzureOpenAI]


sync_client = AzureOpenAI(
    api_version="2023-07-01",
    api_key="example API key",
    azure_endpoint="https://example-resource.azure.openai.com",
)

async_client = AsyncAzureOpenAI(
    api_version="2023-07-01",
    api_key="example API key",
    azure_endpoint="https://example-resource.azure.openai.com",
)


@pytest.mark.parametrize(
    "client,method",
    [
        (sync_client, "copy"),
        (sync_client, "with_options"),
        (async_client, "copy"),
        (async_client, "with_options"),
    ],
)
def test_client_copying(client: Client, method: Literal["copy", "with_options"]) -> None:
    if method == "copy":
        copied = client.copy()
    else:
        copied = client.with_options()

    assert copied._custom_query == {"api-version": "2023-07-01"}


@pytest.mark.parametrize(
    "client",
    [sync_client, async_client],
)
def test_client_copying_override_options(client: Client) -> None:
    copied = client.copy(
        api_version="2022-05-01",
    )
    assert copied._custom_query == {"api-version": "2022-05-01"}


@pytest.mark.parametrize(
    "client,base_url,api_path,json_data,expected",
    [
        (
            AzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                azure_endpoint="https://example-resource.azure.openai.com",
            ),
            "https://example-resource.azure.openai.com/openai/",
            "/chat/completions",
            {"model": "my-deployment"},
            "https://example-resource.azure.openai.com/openai/deployments/my-deployment/chat/completions?api-version=2024-02-01"
        ),
        (
            AzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                azure_endpoint="https://example-resource.azure.openai.com",
            ),
            "https://example-resource.azure.openai.com/openai/",
            "/models",
            {},
            "https://example-resource.azure.openai.com/openai/models?api-version=2024-02-01"
        ),
        (
            AzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                azure_endpoint="https://example-resource.azure.openai.com",
                azure_deployment="my-deployment"
            ),
            "https://example-resource.azure.openai.com/openai/deployments/my-deployment/",
            "/chat/completions",
            {"model": "placeholder"},
            "https://example-resource.azure.openai.com/openai/deployments/my-deployment/chat/completions?api-version=2024-02-01"
        ),
        (
            AzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                azure_endpoint="https://example-resource.azure.openai.com",
                azure_deployment="my-deployment"
            ),
            "https://example-resource.azure.openai.com/openai/deployments/my-deployment/",
            "/models",
            {},
            "https://example-resource.azure.openai.com/openai/models?api-version=2024-02-01"
        ),
        (
            AzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                base_url="https://example.azure-api.net/PTU/deployments/my-deployment/",
            ),
            "https://example.azure-api.net/PTU/deployments/my-deployment/",
            "/chat/completions",
            {"model": "placeholder"},
            "https://example.azure-api.net/PTU/deployments/my-deployment/chat/completions?api-version=2024-02-01"
        ),
        (
            AzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                base_url="https://example.azure-api.net/PTU/deployments/my-deployment/",
            ),
            "https://example.azure-api.net/PTU/deployments/my-deployment/",
            "/models",
            {},
            "https://example.azure-api.net/PTU/models?api-version=2024-02-01"
        ),
        (
            AsyncAzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                azure_endpoint="https://example-resource.azure.openai.com",
            ),
            "https://example-resource.azure.openai.com/openai/",
            "/chat/completions",
            {"model": "my-deployment"},
            "https://example-resource.azure.openai.com/openai/deployments/my-deployment/chat/completions?api-version=2024-02-01"
        ),
        (
            AsyncAzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                azure_endpoint="https://example-resource.azure.openai.com",
            ),
            "https://example-resource.azure.openai.com/openai/",
            "/models",
            {},
            "https://example-resource.azure.openai.com/openai/models?api-version=2024-02-01"
        ),
        (
            AsyncAzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                azure_endpoint="https://example-resource.azure.openai.com",
                azure_deployment="my-deployment"
            ),
            "https://example-resource.azure.openai.com/openai/deployments/my-deployment/",
            "/chat/completions",
            {"model": "placeholder"},
            "https://example-resource.azure.openai.com/openai/deployments/my-deployment/chat/completions?api-version=2024-02-01"
        ),
        (
            AsyncAzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                azure_endpoint="https://example-resource.azure.openai.com",
                azure_deployment="my-deployment"
            ),
            "https://example-resource.azure.openai.com/openai/deployments/my-deployment/",
            "/models",
            {},
            "https://example-resource.azure.openai.com/openai/models?api-version=2024-02-01"
        ),
        (
            AsyncAzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                base_url="https://example.azure-api.net/PTU/deployments/my-deployment/",
            ),
            "https://example.azure-api.net/PTU/deployments/my-deployment/",
            "/chat/completions",
            {"model": "placeholder"},
            "https://example.azure-api.net/PTU/deployments/my-deployment/chat/completions?api-version=2024-02-01"
        ),
        (
            AsyncAzureOpenAI(
                api_version="2024-02-01",
                api_key="example API key",
                base_url="https://example.azure-api.net/PTU/deployments/my-deployment/",
            ),
            "https://example.azure-api.net/PTU/deployments/my-deployment/",
            "/models",
            {},
            "https://example.azure-api.net/PTU/models?api-version=2024-02-01"
        ),
    ],
)
def test_client_prepare_url(client: Client, base_url: str, api_path: str, json_data: dict[str, str], expected: str) -> None:
    req = client._build_request(
        FinalRequestOptions.construct(
            method="post",
            url=api_path,
            json_data=json_data,
        )
    )
    assert req.url == expected
    assert client.base_url == base_url
