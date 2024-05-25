from dotenv import load_dotenv
from django.conf import settings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .credentials import ClientCredentials
import uuid
import re


load_dotenv()


class LLMHybridRetriever(ClientCredentials):

    def __init__(self, alpha: float = 0.8, num_results: int = 45, verbose: bool = False):
        super(LLMHybridRetriever, self).__init__()

        self.alpha = alpha
        self.num_results = num_results
        self.verbose = verbose

        self.__llm_class = ChatOpenAI(openai_api_key=settings.OPENAI_API_KEY,
                                      model_name=settings.GPT_MODEL,
                                      openai_api_base=settings.BASE_URL,
                                      temperature=0.4,
                                      max_tokens=200)

        if verbose is True:
            print(f"WEAVIATE CONNECTION STABLE: {self.weaviate_client.is_live()}")

        self.__PROMPT_CLASS = """Be a helpful assistant. Classify the given user prompt into the following category. Do not provide an explanation or a long response. Just give the category. No need to correct spelling mistakes for users. No need to provide extra information. Return category only. Do not provide any answer outside these categories.

Named Concept(s) or Framework(s) or Method(s) or Model(s)
Organizational Change or Organizational Management
Opinion or Subjective
Actionable Requests
Greeting
Definitional Questions
Specific Domain Knowledge
General Knowledge
Context Required
Ambiguous


USER PROMPT: {user_prompt}"""

        self.__prompt_class = PromptTemplate.from_template(self.__PROMPT_CLASS)

        self.__chain_class = LLMChain(llm=self.__llm_class, prompt=self.__prompt_class)

    # Comprehensive phrase

    def trigger_vectors(self, query: str) -> bool:
        cat = self.__chain_class.run(user_prompt=query)

        if self.verbose is True:
            print(f"QUESTION CATEGORY: {cat}")

        if "Ambiguous" == cat or "Definitional Questions" == cat \
                or "Named Concept(s), Framework(s), Method(s), Model(s)" in cat \
                or "Specific Domain Knowledge" in cat \
                or "Organizational Change or Organizational Management" in cat \
                or "Opinion or Subjective" in cat:

            if "Ambiguous" in cat and "Context Required" in cat:
                return False
            else:
                return True

    def collection_exists(self, class_: str) -> bool:
        return True if self.weaviate_client.schema.exists(class_) is True else False

    def add_collection(self, class_: str) -> None:
        class_obj = {
            "class": class_,
            "description": f"collection for {class_}",
            "vectorizer": "text2vec-cohere",
            "properties": [
                {
                    "name": "uuid",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "skip": True,
                        }
                    }
                },
                {
                    "name": "entity",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "skip": True,
                        }
                    }
                },

                {
                    "name": "user_id",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "skip": True,

                        }
                    }
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "vectorizePropertyName": True,
                            "model": "embed-multilingual-v3.0",
                        }
                    }
                }
            ],
        }

        self.weaviate_client.schema.create_class(class_obj)

    def search_vectors_user(self, query: str, class_: str, entity: str, user_id: str, show_score: bool = False) -> list:
        try:
            weaviate_result = []

            filter_query = re.sub(r"\\", "", query).lower()

            response = (
                self.weaviate_client.query
                .get(class_, ["content"])
                .with_where(
                    {
                        "operator": "And",
                        "operands": [
                            {
                                "path": ["entity"],
                                "operator": "Equal",
                                "valueText": entity,
                            },
                            {
                                "path": ["user_id"],
                                "operator": "Equal",
                                "valueText": user_id,
                            },
                        ]
                    }
                )

                .with_hybrid(
                    query=filter_query,
                    alpha=self.alpha,
                )
                # .with_additional("score")
                .with_limit(self.num_results)
                .do()
            )

            if not response or response != [] or 'data' in response:
                print("FOUND VECTORS!")
                result = response['data']['Get'][class_]

                if result is not None:
                    for chunk in result:
                        weaviate_result.append(chunk['content'])

                return weaviate_result
            else:
                print("NO RESULTS FOUND!")
                return []
        except Exception as e:
            print(f"CLASS LLMHYBRID -> USER VEC: {e}")
            return []

    def search_vectors_initiative(self, query: str, class_: str, entity: str) -> list:
        try:
            weaviate_result = []

            filter_query = re.sub(r"\\", "", query).lower()

            response = (
                self.weaviate_client.query
                .get(class_, ["content"])
                .with_where(
                    {
                        "path": ["entity"],
                        "operator": "Equal",
                        "valueText": entity,
                    }
                )

                .with_hybrid(
                    query=filter_query,
                    alpha=self.alpha,
                )
                # .with_additional("score")
                .with_limit(self.num_results)
                .do()
            )

            if not response or response != [] or 'data' in response:
                print("FOUND VECTORS!")
                result = response['data']['Get'][class_]
                if result is not None:
                    for chunk in result:
                        weaviate_result.append(chunk['content'])

                return weaviate_result
            else:
                print("NO RESULTS FOUND!")
                return []
        except Exception as e:
            print(f"CLASS LLMHYBRID -> INIT VEC: {e}")
            return []

    def search_vectors_company(self, query: str, class_: str, entity: str) -> list:
        try:
            weaviate_result = []

            filter_query = re.sub(r"\\", "", query).lower()

            response = (
                self.weaviate_client.query
                .get(class_, ["content"])
                .with_where(
                    {
                        "path": ["entity"],
                        "operator": "Equal",
                        "valueText": entity,
                    }
                )

                .with_hybrid(
                    query=filter_query,
                    alpha=self.alpha,
                )
                # .with_additional("score")
                .with_limit(self.num_results)
                .do()
            )

            if not response or response != [] or 'data' in response:
                print("FOUND VECTORS!")
                result = response['data']['Get'][class_]
                if result is not None:
                    for chunk in result:
                        weaviate_result.append(chunk['content'])

                return weaviate_result
            else:
                print("NO RESULTS FOUND!")
                return []
        except Exception as e:
            print(f"CLASS LLMHYBRID -> COMP VEC: {e}")
            return []

    def reranker(self, query: str, batch: list, top_k: int = 6, return_type: type = str) -> str or list:
        try:
            if not batch:
                return "\n\n".join(batch) if return_type == str else batch

            ranked_results = []

            results = self.cohere_client.rerank(
                query=query,
                documents=batch,
                top_n=top_k,
                model='rerank-english-v2.0',
                return_documents=True)

            for document in results.results:
                ranked_results.append(document.document.text)

            return "\n\n".join(ranked_results) if return_type == str else ranked_results
        except Exception as e:
            print(e)
            return [] if return_type == list else ""

    def add_batch(self, batch: list, user_id: str, entity: str, class_: str) -> str:
        try:
            unique_id = uuid.uuid4()

            data_objs = [
                {
                    "entity": str(entity),
                    "user_id": str(user_id),
                    "content": str(chunk),
                    "uuid": str(unique_id)
                } for chunk in batch
            ]

            self.weaviate_client.batch.configure(batch_size=100)
            with self.weaviate_client.batch as batch_data:
                for data_obj in data_objs:
                    batch_data.add_data_object(
                        data_obj,
                        class_,
                    )

            return str(unique_id)
        except Exception as e:
            print(e)
            return ""

    def filter_by(self, key: str, value: str, class_: str):
        response = (
            self.weaviate_client.query
            .get(class_, ["content"])
            .with_where({
                'path': [key],
                'operator': 'Equal',
                'valueText': value
            })

            .with_additional("id")
            .with_limit(self.num_results)
            .do()
        )

        result = response['data']['Get'][class_]

        return result

    def get_by_id(self, class_: str, obj_id: str):
        data_object = self.weaviate_client.data_object.get_by_id(
            obj_id,
            class_name=class_,
        )

        return data_object
