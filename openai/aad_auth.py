import openai
import logging
import typing
import requests
import aiohttp
import asyncio
import time
if typing.TYPE_CHECKING:
    from azure.core.credentials import TokenCredential
    from azure.core.credentials_async import AsyncTokenCredential
log = logging.getLogger(__name__)


class TokenCredentialAuth:

    def __init__(self, credential, *scopes):
        self.credential = credential
        self.scopes = [scopes] if isinstance(scopes, str) else scopes
        self.cached_token: str | None = None

    def __call__(self, r):
        if not self.cached_token or self.cached_token.expires_on - time.time() < 300:
            self.cached_token = self.credential.get_token(*self.scopes)
        r.headers["Authorization"] = f"Bearer " + self.cached_token.token
        print("adding auth header")
        return r


class AsyncTokenCredentialAuth(aiohttp.ClientRequest):

    async def _refresh_token(self):
        if self.auth.get("cached_token") is None or self.auth.get("cached_token").expires_on - time.time() < 300:
            async with self.auth.get("lock"):
                if self.auth.get("cached_token") is None or self.auth.get("cached_token").expires_on - time.time() < 300:
                    print("i called get_token")
                    self.auth["cached_token"] = await self.auth["credential"].get_token(*self.auth.get("scopes"))
        self.headers["Authorization"] = "Bearer " + self.auth["cached_token"].token

    async def send(self, conn: "Connection") -> "ClientResponse":
        await self._refresh_token()
        return await super().send(conn)


def login(
    endpoint: str, credential: "TokenCredential", *, api_version: str | None = None
) -> None:
    if openai.api_version and api_version:
        log.info(f'Overriding openai.api_version "{openai.api_version}" with api_version "{api_version}" passed to easyaz.openai.login') 
    openai.api_version = api_version or openai.api_version or '2022-12-01'
    
    if openai.api_base and endpoint != openai.api_base:
        log.info(f'Overriding openai.endpoint "{openai.api_base}" with endpoint "{endpoint}" passed to easyaz.openai.login')
    openai.api_base = endpoint

    openai.api_key = "AZUREAD_FAKE_API_KEY"
    openai.api_type = "azuread"

    def factory():
        session = requests.Session()
        session.auth = TokenCredentialAuth(credential, "https://cognitiveservices.azure.com/.default")
        return session

    openai.session_factory = factory
    request_class = AsyncTokenCredentialAuth
    request_class.auth = {"credential": credential, "scopes": ["https://cognitiveservices.azure.com/.default"], "cached_token": None, "lock": asyncio.Lock()}
    sess = aiohttp.ClientSession(request_class=AsyncTokenCredentialAuth)
    openai.aiosession.set(sess)
    return sess


# from azure.identity import DefaultAzureCredential
# import openai
# from multiprocessing.pool import ThreadPool
# # openai.api_type = "azure"

# openai.api_base = "https://openai-shared.openai.azure.com"
# openai.api_version = "2023-03-15-preview"
# credential = DefaultAzureCredential()
# login(endpoint="https://openai-shared.openai.azure.com", credential=credential)
# openai.api_key = credential.get_token().token
# completion = openai.Completion.create(prompt="hello", engine="text-davinci-003")

# def task(item):
    
#     completion = openai.Completion.create(prompt=item, engine="text-davinci-003")
#     return completion

# items = ["test", "hello world", "does this work?", "what's your name?"]
# with ThreadPool(3) as pool:
#     for result in pool.map(task, items):
#         print(result)




import pytest

@pytest.mark.asyncio
async def test_it():
    from azure.identity.aio import DefaultAzureCredential
    import time
    
    openai.api_type = "azuread"
    openai.api_base = "https://openai-shared.openai.azure.com"
    openai.api_version = "2023-03-15-preview"


    credential = DefaultAzureCredential()
    async with login(endpoint="https://openai-shared.openai.azure.com", credential=credential):
        async def openaicall():
            completion = await openai.Completion.acreate(prompt="hello world", engine="text-davinci-003")
            return completion
        result = await asyncio.gather(
            openaicall(),
            openaicall(),
            openaicall(),
        )
        print(result)
        await credential.close()

asyncio.run(test_it())
