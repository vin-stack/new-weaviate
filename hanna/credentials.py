import cohere
import weaviate
from django.conf import settings
from dotenv import load_dotenv
import base64

load_dotenv()
username = "root"
password = "CZLTVzJz-PgmQ-jttFcU9k"

class ClientCredentials:

    def __init__(self):
        self.cohere_client = cohere.Client(settings.COHERE_API_KEY)
        self.__auth_config = weaviate.auth.AuthApiKey(api_key=settings.WEAVIATE_API_KEY)
        self.weaviate_client = weaviate.Client(
            url=settings.WEAVIATE_URL,
            additional_headers={"X-Cohere-Api-Key": settings.COHERE_API_KEY},
            auth_client_secret=self.__auth_config
        )
