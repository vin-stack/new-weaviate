from channels.generic.websocket import AsyncWebsocketConsumer
from channels.exceptions import RequestTimeout
from channels.db import database_sync_to_async
from .models import UserPrompts, SystemSettings
import json
import os
import cohere
from dotenv import load_dotenv
import weaviate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import re
import asyncio
import fasttext
from django.conf import settings

load_dotenv()


co = cohere.Client(os.environ.get("COHERE_API_KEY"))

auth_config = weaviate.auth.AuthApiKey(api_key=os.environ.get("WEAVIATE_API_KEY")) #os.environ.get("WEAVIATE_API_KEY"))
WEAVIATE_URL = os.environ.get("WEAVIATE_URL")
headers = {"X-Cohere-Api-Key": os.environ.get("COHERE_API_KEY")}


client = weaviate.Client(
    url=WEAVIATE_URL,
    additional_headers=headers,
    auth_client_secret=auth_config
)

OPENAI_API_KEY = os.environ.get('DEEPINFRA')
GPT_MODEL_1 = "mistralai/Mixtral-8x7B-Instruct-v0.1"
GPT_MODEL_2 = "mistralai/Mixtral-8x22B-Instruct-v0.1"
BASE_URL = "https://api.deepinfra.com/v1/openai"

job_done = "STOP"


class SimpleCallback(BaseCallbackHandler):
    def __init__(self, q: asyncio.Queue):
        self.q = q

    async def on_llm_start(self, serialized, prompts, **kwargs):
        if settings.DEBUG is True:
            print(f"The LLM has Started")

        while not self.q.empty():
            try:
                await self.q.get()
            except asyncio.Queue.empty:
                continue

    async def on_llm_new_token(self, token: str, *, chunk=None, run_id, parent_run_id=None, **kwargs):
        await self.q.put(chunk.text)
        # print(chunk.text)

    async def on_llm_end(self, *args, **kwargs):
        await self.q.put(job_done)
        if settings.DEBUG is True:
            print("The LLM has ended!")


def hybrid_retriever(query, alpha=0.8, num_results=60, top_n=10):
    llm_class = ChatOpenAI(openai_api_key=OPENAI_API_KEY,
                           model_name=GPT_MODEL_1,
                           openai_api_base=BASE_URL)

    # Definitional Questions adjective
    # Definitional Questions noun

    PROMPT_CLASS = """Be a helpful assistant. Classify the given user prompt into the following category. Do not provide an explanation or a long response. Just give the category. No need to correct spelling mistakes for users. No need to provide extra information. Return category only. Do not provide any answer outside these categories.

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

    # Comprehensive phrase

    prompt_class = PromptTemplate.from_template(PROMPT_CLASS)

    chain_class = LLMChain(llm=llm_class, prompt=prompt_class)

    cat = chain_class.run(user_prompt=query)

    if settings.DEBUG is True:
        print(f"QUESTION CATEGORY: {cat}")

    if "Ambiguous" == cat or "Definitional Questions" == cat \
            or "Named Concept(s) or Framework(s) or Method(s) or Model(s)" in cat \
            or "Specific Domain Knowledge" in cat \
            or "Organizational Change or Organizational Management" in cat \
            or "Opinion or Subjective" in cat:
        if "Ambiguous" in cat and "Context Required" in cat:
            return ""
        else:
            print("GETTING VECTORS!")
            filter_query = re.sub(r"\\", "", query)

            response = (
                client.query
                .get("EAM", ["text"])
                .with_hybrid(
                    query=filter_query,
                    alpha=alpha,
                )
                .with_additional("score")
                .with_limit(num_results)
                .do()
            )

            result = response['data']['Get']['EAM']

            ranked_result = []

            if result is not None:
                results = co.rerank(query=query, documents=result, top_n=top_n, model='rerank-english-v2.0')

                print(results.results)

                for chunk in results.results:
                    ranked_result.append(chunk.document['text'])

                return ranked_result

            else:
                return ""
    else:
        return ""


class ChatConsumer(AsyncWebsocketConsumer):
    def getSystemSettings(self, param) -> str:
        return SystemSettings.objects.get(param=param).description

    def displayChatHistory(self, email) -> None:
        UserPrompts.objects.filter(email=email).filter(displayed=False).update(displayed=True)
        return None

    def windowBufferMemory(self, email, k=7):
        tmp = UserPrompts.objects.order_by('-date').filter(email=email).filter(displayed=False)[:k]
        history = {"history": ""}

        print("\n\nMEMORY:")
        for mem in reversed(tmp):
            print(f"Username: {mem.name} PROMPT: {mem.prompt}")
        print("END MEMORY\n\n")

        chats = reversed([f"\n\n[INST]{t.prompt}[/INST]\n{t.response}" for t in tmp])
        for chat in chats:
            history['history'] += chat
        return history

    def detectLanguage(self, text, original_language):
        # Load the pre-trained language identification model
        model_path = 'lid.176.ftz'  # Path to the pre-trained model file
        model = fasttext.load_model(model_path)

        # Text to be identified
        text = text.strip()

        # If the text is less than 5 words, return the original language
        if len(text.split()) < 5:
            return original_language

        # Predict the language of the text
        predicted_languages = model.predict(text, k=1)  # Get top 1 prediction
        detected_language_code = predicted_languages[0][0].replace('__label__', '')

        # Mapping of language codes to language names. I think with these languages we have more than enough
        language_names = {
            'en': 'English',
            'es': 'Spanish',
            'fr': 'French',
            'de': 'German',
            'it': 'Italian',
            'pt': 'Portuguese',
            'zh': 'Chinese',
            'ru': 'Russian',
            'ja': 'Japanese',
            'nl': 'Dutch',
            'ar': 'Arabic',
            'ur': 'Urdu'
        }

        # Return the full name of the detected language
        return language_names.get(detected_language_code, original_language)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.systemPrompt = ""
        self.prompt = ""
        self.tmp_mem = ""
        self.que = asyncio.Queue()
        self.llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            model_name=GPT_MODEL_2,
            openai_api_base=BASE_URL,
            streaming=True,
            max_tokens=1000,
            callbacks=[SimpleCallback(self.que)])

    @database_sync_to_async
    def save_prompt(self, name: str, email: str, user_prompt: str, system_response: str, special_instruction: str):
        if settings.DEBUG is True:
            print(f"MESSAGE SAVED!, NAME: {name}")
        UserPrompts.objects.create(
            name=name,
            email=email,
            prompt=user_prompt,
            response=system_response,
            special_instruction=special_instruction)

    async def process_question(self, question, username, email, language):
        remove_tag = ""
        sp = ""

        if "/opinion" in question:
            print("Asking for opinion!!!")

            remove_tag = re.sub("/opinion", "", question)
            print(remove_tag)
            temp = await database_sync_to_async(self.getSystemSettings)("/opinion")
            sp = f"""\n{temp} \n{remove_tag}"""

        else:
            remove_tag = question
            sp = question

        retriever = hybrid_retriever(remove_tag)
        chain = LLMChain(llm=self.llm, prompt=self.prompt)
        memory = await database_sync_to_async(self.windowBufferMemory)(email=email)
        await chain.arun(
            matching_model='\n'.join(retriever),
            username=username,
            question=sp,
            chat_history=memory['history'],
            language_to_use=language
        )

    async def generate_response(self, name, email, query):
        remove_tag = ""
        txt = ""
        while True:
            next_token = await self.que.get()  # Blocks until an input is available
            if next_token is job_done:
                await self.send(text_data=json.dumps({"msg": job_done}))
                break
            txt += next_token
            await self.send(text_data=json.dumps({"msg": next_token}))
            await asyncio.sleep(0.01)
            self.que.task_done()

        if "/opinion" in query:
            remove_tag = re.sub("/opinion", "", query)
        else:
            remove_tag = query

        matches = re.findall(r'\/\w+', query)
        tag = ""
        if len(matches) == 0:
            tag = ""
        else:
            tag = matches[0]
        await self.save_prompt(name=name, email=email, user_prompt=remove_tag, system_response=txt, special_instruction=tag)

    async def start(self, query, name, email, language):
        task_1 = asyncio.create_task(self.process_question(question=query, email=email, username=name, language=language))
        task_2 = asyncio.create_task(self.generate_response(name=name, email=email, query=query))

        await task_1
        await task_2

    async def connect(self):
        await self.accept()

        self.llm.temperature = 0.4

        temp = await database_sync_to_async(self.getSystemSettings)("system-prompt")
        passage = temp
        self.prompt = PromptTemplate(
            input_variables=["chat_history", "question", "matching_model"], template=passage
        )

    async def receive(self, text_data=None, byte_data=None):

        try:
            jdata = json.loads(text_data)

            if settings.DEBUG is True:
                print(jdata)

            await database_sync_to_async(self.windowBufferMemory)(email=jdata['email'])

            if 'history_clear' in jdata:
                await database_sync_to_async(self.displayChatHistory)(email=jdata['email'])
                await self.send(text_data=json.dumps({"history": "history is cleared!", "tmp": self.tmp_mem}))

            if 'query' in jdata:
                if jdata['query'] == "":
                    await self.send(text_data=json.dumps({"error": "Please fill in the required field!"}))

                if len(jdata['query']) < 801:
                    lang = self.detectLanguage(text=jdata['query'], original_language=jdata['language'])
                    print("DETECTED LANGUAGE: ", lang)

                    # if jdata['language'] == "SPANISH":
                    #     set_lang = "Reply in SPANISH and dont use any other language or translation. "
                    #     jdata['query'] = set_lang + jdata['query']

                    await self.start(query=jdata['query'], name=jdata['name'], email=jdata['email'], language=lang)
                else:
                    await self.send(text_data=json.dumps(
                        {"error": "Sorry but you are only allowed to send prompts that are 800 characters long!"}))

        except Exception as e:
            print(e)
            await self.send(text_data=json.dumps({"error": "I’m having some trouble processing your request. I will record the error and inform our support team. Thank you very much! Hanna"}))

        except RequestTimeout as e:
            print(e)

    # async def disconnect(self, code):
    #     print("DISCONNECTED")
