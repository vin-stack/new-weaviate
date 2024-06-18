import asyncio
from dotenv import load_dotenv
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .credentials import ClientCredentials
import uuid
import re
from weaviate.gql.get import HybridFusion

load_dotenv()

class LLMHybridRetriever(ClientCredentials):

    def __init__(self, alpha: float = 0.7, num_results: int = 45, verbose: bool = False):
        super(LLMHybridRetriever, self).__init__()

        self.alpha = alpha
        self.num_results = num_results
        self.verbose = verbose

        self.__llm_class = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.GPT_MODEL_2,
            openai_api_base=settings.BASE_URL,
            temperature=0.4,
            max_tokens=200
        )
        self.threshold = 0.65

        if verbose:
            print(f"WEAVIATE CONNECTION STABLE: {self.weaviate_client.is_live()}")

        self.__PROMPT_CLASS = """<s>[INST] You are a Classifier Algorithm..."""
        self.__prompt_class = PromptTemplate.from_template(self.__PROMPT_CLASS)
        self.__chain_class = LLMChain(llm=self.__llm_class, prompt=self.__prompt_class)

    async def trigger_vectors(self, query: str) -> bool:
        cat = await asyncio.to_thread(self.__chain_class.run, user_prompt=query)
        if self.verbose:
            print(f"QUESTION CATEGORY: {cat}")
        return cat

    async def collection_exists(self, class_: str) -> bool:
        return await asyncio.to_thread(self.weaviate_client.schema.exists, class_)

    def add_collection(self, class_: str) -> None:
        class_obj = {
            "class": class_,
            "description": f"collection for {class_}",
            "vectorizer": "text2vec-cohere",
            "properties": [
                {
                    "name": "uuid",
                    "dataType": ["text"],
                    "moduleConfig": {"text2vec-cohere": {"skip": True}}
                },
                {"name": "entity", "dataType": ["text"], "moduleConfig": {"text2vec-cohere": {"skip": True}}},
                {"name": "user_id", "dataType": ["text"], "moduleConfig": {"text2vec-cohere": {"skip": True}}},
                {
                    "name": "content",
                    "dataType": ["text"],
                    "moduleConfig": {"text2vec-cohere": {"vectorizePropertyName": True, "model": "embed-multilingual-v3.0"}}
                }
            ],
        }
        self.weaviate_client.schema.create_class(class_obj)

    async def search_vectors_user(self, query: str, class_: str, entity: str, user_id: str, show_score: bool = False) -> list:
        try:
            weaviate_result = []
            filter_query = re.sub(r"\\", "", query).lower()

            response = await asyncio.to_thread(
                self.weaviate_client.query
                .get(class_, ["content"])
                .with_where({"operator": "And", "operands": [{"path": ["entity"], "operator": "Equal", "valueText": entity}, {"path": ["user_id"], "operator": "Equal", "valueText": user_id}]})
                .with_hybrid(query=filter_query, alpha=self.alpha, fusion_type=HybridFusion.RELATIVE_SCORE)
                .with_additional("score")
                .with_limit(self.num_results)
                .do
            )

            if response and 'data' in response:
                result = response['data']['Get'][class_]
                if result:
                    for chunk in result:
                        relevance_score = round(float(chunk['_additional']['score']), 3)
                        if relevance_score >= self.threshold:
                            weaviate_result.append(chunk['content'])
                return weaviate_result
            return []
        except Exception as e:
            print(f"CLASS LLMHYBRID -> USER VEC: {e}")
            return []

    async def search_vectors_initiative(self, query: str, class_: str, entity: str) -> list:
        try:
            weaviate_result = []
            filter_query = re.sub(r"\\", "", query).lower()

            response = await asyncio.to_thread(
                self.weaviate_client.query
                .get(class_, ["content"])
                .with_where({"path": ["entity"], "operator": "Equal", "valueText": entity})
                .with_hybrid(query=filter_query, alpha=self.alpha, fusion_type=HybridFusion.RELATIVE_SCORE)
                .with_additional("score")
                .with_limit(self.num_results)
                .do
            )

            if response and 'data' in response:
                result = response['data']['Get'][class_]
                if result:
                    for chunk in result:
                        relevance_score = round(float(chunk['_additional']['score']), 3)
                        if relevance_score >= self.threshold:
                            weaviate_result.append(chunk['content'])
                return weaviate_result
            return []
        except Exception as e:
            print(f"CLASS LLMHYBRID -> INIT VEC: {e}")
            return []

    async def search_vectors_company(self, query: str, class_: str, entity: str) -> list:
        try:
            weaviate_result = []
            filter_query = re.sub(r"\\", "", query).lower()

            response = await asyncio.to_thread(
                self.weaviate_client.query
                .get(class_, ["content"])
                .with_where({"path": ["entity"], "operator": "Equal", "valueText": entity})
                .with_hybrid(query=filter_query, alpha=self.alpha, fusion_type=HybridFusion.RELATIVE_SCORE)
                .with_additional("score")
                .with_limit(self.num_results)
                .do
            )

            if response and 'data' in response:
                result = response['data']['Get'][class_]
                if result:
                    for chunk in result:
                        relevance_score = round(float(chunk['_additional']['score']), 3)
                        if relevance_score >= self.threshold:
                            weaviate_result.append(chunk['content'])
                return weaviate_result
            return []
        except Exception as e:
            print(f"CLASS LLMHYBRID -> COMP VEC: {e}")
            return []

    async def reranker(self, query: str, batch: list, top_k: int = 6, return_type: type = str) -> str or list:
        try:
            if not batch:
                return "\n\n".join(batch) if return_type == str else batch

            ranked_results = []

            results = await asyncio.to_thread(
                self.cohere_client.rerank,
                query=query,
                documents=batch,
                top_n=top_k,
                model='rerank-english-v2.0',
                return_documents=True
            )

            for document in results.results:
                if float(document.relevance_score) >= self.threshold:
                    ranked_results.append(document.document.text)

            return "\n\n".join(ranked_results) if return type == str else ranked_results
        except Exception as e:
            print(e)
            return [] if return_type == list else ""

    async def add_batch(self, batch: list, user_id: str, entity: str, class_: str) -> str:
        try:
            unique_id = uuid.uuid4()

            data_objs = [{"entity": str(entity), "user_id": str(user_id), "content": str(chunk), "uuid": str(unique_id)} for chunk in batch]

            self.weaviate_client.batch.configure(batch_size=100)
            await asyncio.to_thread(
                self.weaviate_client.batch.add_data_object,
                data_objs,
                class_
            )

            return str(unique_id)
        except Exception as e:
            print(e)
            return ""

    async def add_batch_uuid(self, batch: list, user_id: str, entity: str, uuid_: str, class_: str) -> str:
        try:
            data_objs = [{"entity": str(entity), "user_id": str(user_id), "content": str(chunk), "uuid": str(uuid_)} for chunk in batch]

            self.weaviate_client.batch.configure(batch_size=100)
            await asyncio.to_thread(
                self.weaviate_client.batch.add_data_object,
                data_objs,
                class_
            )

            return str(uuid_)
        except Exception as e:
            print(e)
            return ""

    async def get_by_uuid(self, uuid_: str, class_: str) -> str:
        try:
            response = await asyncio.to_thread(
                self.weaviate_client.query
                .get(class_, ["uuid"])
                .with_additional("id")
                .with_limit(self.num_results)
                .do
            )

            if response and 'data' in response:
                result = response['data']['Get'][class_][0]['_additional']['id']
                return result
            return ""
        except Exception as e:
            print("FILTER BY FUNCTION!")
            print(e)
            return ""

    async def check_uuid(self, uuid: str, class_: str) -> bool:
        try:
            await asyncio.to_thread(
                self.weaviate_client.data_object.update,
                uuid=uuid,
                class_name=class_,
                data_object={"type": 100}
            )
            return True
        except Exception as e:
            print("UUID CHECK: ", e)
            return False
    async def get_by_id(self, class_: str, obj_id: str):
        try:
            data_object = await asyncio.to_thread(
                self.weaviate_client.data_object.get_by_id,
                obj_id,
                class_name=class_
            )
            return data_object
        except Exception as e:
            print(f"Error in get_by_id: {e}")
            return None

