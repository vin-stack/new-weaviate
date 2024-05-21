from cohere import RerankResponseResultsItem
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

    def __init__(self, alpha: float = 0.8, num_results: int = 60, top_n: int = 6, verbose: bool = False):
        super(LLMHybridRetriever, self).__init__()

        self.alpha = alpha
        self.num_results = num_results
        self.top_n = top_n
        self.verbose = verbose

        self.__llm_class = ChatOpenAI(openai_api_key = settings.OPENAI_API_KEY,
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

    def search(self, query, class_: str, entity: str, user: str):
        cat = self.__chain_class.run(user_prompt=query)

        if self.verbose is True:
            print(f"QUESTION CATEGORY: {cat}")

        if "Ambiguous" == cat or "Definitional Questions" == cat \
                or "Named Concept(s), Framework(s), Method(s), Model(s)" in cat \
                or "Specific Domain Knowledge" in cat \
                or "Organizational Change or Organizational Management" in cat \
                or "Opinion or Subjective" in cat:
            if "Ambiguous" in cat and "Context Required" in cat:
                return ""
            else:
                filter_query = re.sub(r"\\", "", query)

                response = (
                    self.weaviate_client.query
                    .get(class_, ["content"])
                    .with_where({
                        'path': ['entity'],
                        'operator': 'Equal',
                        'valueText': entity
                    })
                    .with_hybrid(
                        query=filter_query,
                        alpha=self.alpha,
                    )
                    # .with_additional("score")
                    .with_limit(self.num_results)
                    .do()
                )

                result = response['data']['Get'][class_]
                weaviate_result = []
                ranked_result = []

                if result is not None:
                    for chunk in result:
                        weaviate_result.append(chunk['content'])

                    results = self.cohere_client.rerank(
                        query=query,
                        documents=weaviate_result,
                        top_n=self.top_n,
                        model='rerank-english-v2.0',
                        return_documents=True)
                    
                    company_weight = 0.3
                    entity_weight = 0.3
                    user_weight = 0.3
                    update_results=[]
                    for document in results.results:
                        relevance_score = document.relevance_score
                        overall_score = (company_weight * relevance_score) + (entity_weight * relevance_score) + (user_weight * relevance_score)
                        doc = RerankResponseResultsItem(index=document.index, relevance_score=relevance_score, overall_score = overall_score, text=document.document.text if hasattr(document,'document') and hasattr(document.document, 'text') else None)
                        update_results.append(doc)
                    update_results.sort(key=lambda x: x.overall_score, reverse=True)
                    
                    for chunk in update_results:
                        ranked_result.append(chunk.text)

                    return "\n".join(ranked_result)
                else:
                    return ""
        else:
            return ""

    def add_batch(self, batch: list, user_id: str, entity: str, class_: str):
        try:
            unique_id = uuid.uuid4()

            data_objs = [
                {
                    "entity": str(entity),
                    "user_id" : str(user_id),
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
