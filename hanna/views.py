import logging
import json
from django.http import StreamingHttpResponse
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from .retriever import LLMHybridRetriever
from .master_vectors.MV import MasterVectors
from .chunker import ChunkText
from dotenv import load_dotenv
from .credentials import ClientCredentials
from langchain.callbacks.tracers import ConsoleCallbackHandler

load_dotenv()

logger = logging.getLogger(__name__)

credentials = ClientCredentials()

with open("test-prompt.txt", "r") as file:
    SYSPROMPT = file.read()

with open("chatnote-prompt.txt", "r") as read_prompt:
    CHATNOTE_PROMPT = read_prompt.read()

prompt = PromptTemplate.from_template(SYSPROMPT)
chat_note_prompt = PromptTemplate.from_template(CHATNOTE_PROMPT)

llm = ChatOpenAI(
    openai_api_key=settings.OPENAI_API_KEY,
    model_name=settings.GPT_MODEL_2,
    openai_api_base=settings.BASE_URL,
    temperature=0.4,
    max_tokens=1000,
    streaming=True
)

llm_hybrid = LLMHybridRetriever(verbose=True)
mv = MasterVectors()
slice_document = ChunkText()

@api_view(http_method_names=['GET'])
def home(request) -> Response:
    return Response({'msg': 'this is hanna enterprise suite'}, status=status.HTTP_200_OK)




@api_view(http_method_names=['POST'])
async def chat_stream(request) -> Response or StreamingHttpResponse:
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])
        query = str(company['query'])
        entity = str(company['entity'])
        user_id = str(company['user_id'])

        if not await llm_hybrid.collection_exists(collection):
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        cat = await llm_hybrid.trigger_vectors(query=query)
        
        master_vector, company_vector, initiative_vector, member_vector = [], [], [], []
        combine_ids = "INP" + entity

        if any(c in cat for c in ["Specific Domain Knowledge", "Organizational Change or Organizational Management", 
                                  "Definitional Questions", "Context Required"]):
            master_vector = await mv.search_master_vectors(query=query, class_="MV001")
            company_vector = await llm_hybrid.search_vectors_company(query=query, entity=collection, class_=collection)
            initiative_vector = await llm_hybrid.search_vectors_initiative(query=query, entity=entity, class_=collection)
            member_vector = await llm_hybrid.search_vectors_user(query=query, class_=collection, entity=combine_ids, user_id=user_id)
        elif any(c in cat for c in ["Individuals", "Personal Information"]):
            company_vector = await llm_hybrid.search_vectors_company(query=query, entity=collection, class_=collection)
            initiative_vector = await llm_hybrid.search_vectors_initiative(query=query, entity=entity, class_=collection)
            member_vector = await llm_hybrid.search_vectors_user(query=query, class_=collection, entity=combine_ids, user_id=user_id)

        initiative_vector.extend(member_vector)
        top_master_vec = await mv.reranker(query=query, batch=master_vector)
        top_company_vec = await llm_hybrid.reranker(query=query, batch=company_vector)
        top_member_initiative_vec = await llm_hybrid.reranker(query=query, batch=initiative_vector, top_k=10)

        retriever = f"{top_master_vec} \n\n {top_company_vec} \n {top_member_initiative_vec}"

        config = {
            'callbacks': [ConsoleCallbackHandler()]
        }

        chain = prompt | llm | StrOutputParser()

        response_stream = chain.stream({
            'matching_model': retriever,
            'question': query,
            'username': company['user'],
            'chat_history': "",
            'language_to_use': company['language']
        }, config=config)

        response = StreamingHttpResponse(response_stream, status=status.HTTP_200_OK, content_type='text/event-stream')
        response['Cache-Control'] = 'no-cache'

        return response
    except Exception as e:
        logger.error(f"Error in chat_stream: {e}")
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    
@api_view(http_method_names=['POST'])
def create_collection(request) -> Response:
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is True:
            return Response({'error': 'This collection already exists!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.add_collection(collection)

        return Response({'msg': f'Collection created!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW CREATE COLLECTION:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def add_vectors(request) -> Response:
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])
        entity = str(company['entity'])
        user_id = str(company['user_id'])
        type_ = str(company['type'])
        print("GOT REQUEST!!!")

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        print("PRE-FILTERING TEXT!")
        pre_filter = str(company['text']).strip().replace("\n", "").lower()

        print("SLICING DOCUMENTS!!!")
        documents = slice_document.chunk_corpus(pre_filter)
        print("DOCUMENTS SLICED!!!")
        # PV -> Private Vec
        # INV -> Init Vec
        # CMV -> Company Vec
        # file -> context -> uuid -> search
        if type_ == "PV":
            # uid = "task started"
            combine_ids = "INP" + entity  # -> Initiative ID
            # uid = call_function.delay(documents, user_id, combine_ids, collection)
            print("SENDInG REQUEST TO WEAVIATE!!!")
            uid = llm_hybrid.add_batch(documents, user_id, combine_ids, collection)

        elif type_ == "CMV":
            # C72 -> entity
            uid = llm_hybrid.add_batch(documents, user_id, collection, collection)
        elif type_ == "INV":
            uid = llm_hybrid.add_batch(documents, user_id, entity, collection)
        else:
            return Response({'error': 'No such type!'}, status=status.HTTP_400_BAD_REQUEST)

        return Response({'msg': str(uid)}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW ADD VEC:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def upload_file(request):
    try:
        data_file = request.FILES['file_upload']
        print(f'File Upload! {data_file}')
        print(request.POST)

        collection = "C" + str(request.POST.get('collection'))
        entity = str(request.POST.get('entity'))
        user_id = str(request.POST.get('user_id'))
        type_ = str(request.POST.get('type'))

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        file_name = data_file.name

        destination = open("./_tmp/" + file_name, 'wb+')
        for chunk in data_file.chunks():
            destination.write(chunk)
        destination.close()

        documents = slice_document.chunk_document(file_name)

        # PV -> Private Vec
        # INV -> Init Vec
        # CMV -> Company Vec
        # file -> context -> uuid -> search
        if type_ == "PV":
            combine_ids = "INP" + entity  # -> Initiative ID
            uid = llm_hybrid.add_batch(documents, user_id, combine_ids, collection)
        elif type_ == "CMV":
            # C72 -> entity
            uid = llm_hybrid.add_batch(documents, user_id, collection, collection)
        elif type_ == "INV":
            uid = llm_hybrid.add_batch(documents, user_id, entity, collection)
        else:
            return Response({'error': 'No such type!'}, status=status.HTTP_400_BAD_REQUEST)

        os.remove("./_tmp/" + file_name)
        return Response({'msg': str(uid)}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def update_chunk(request):
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])
        uuid_ = str(company['uuid'])
        entity = str(company['entity'])
        user_id = str(company['user_id'])
        chunk = str(company['chunk'])
        type_ = str(company['type'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "Equal",
                "valueText": uuid_
            },
        )

        pre_filter = str(chunk).strip().replace("\n", "").lower()

        documents = slice_document.chunk_corpus(pre_filter)

        if type_ == "PV":
            combine_ids = "INP" + entity  # -> Initiative ID
            uid = llm_hybrid.add_batch_uuid(documents, user_id, combine_ids, uuid_, collection)
        elif type_ == "CMV":
            # C72 -> entity
            uid = llm_hybrid.add_batch_uuid(documents, user_id, collection, uuid_, collection)
        elif type_ == "INV":
            uid = llm_hybrid.add_batch_uuid(documents, user_id, entity, uuid_, collection)

        return Response({'msg': 'success'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW UPDATE CHUNK:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_collection(request):
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = (llm_hybrid.weaviate_client.query.get(collection, ['entity', 'uuid', 'content', 'user_id'])
                       .with_additional(["id"])
                       .do())

        res = data_object['data']['Get']
        # print(res)
        return Response({'msg': res}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW GET COLLECTION:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_objects_entity(request):
    try:
        company = json.loads(request.body)
        entity = str(company['entity'])
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = llm_hybrid.filter_by('entity', entity, collection)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW GET OBJECTS ENTITY:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_objects_uuid(request):
    try:
        company = json.loads(request.body)
        uid = str(company['uuid'])
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = llm_hybrid.filter_by('uuid', uid, collection)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW GET OBJECTS UUID:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_object(request):
    try:
        company = json.loads(request.body)
        obj_id = company['id']
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = llm_hybrid.get_by_id(collection, obj_id)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW GET OBJECT:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_object(request):
    try:
        company = json.loads(request.body)
        obj_id = company['id']
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["id"],
                "operator": "Like",
                "valueText": obj_id
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECT:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_objects_entity(request):
    try:
        company = json.loads(request.body)
        entity = str(company['entity'])
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["entity"],
                "operator": "Like",
                "valueText": entity
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECT ENTITY:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_objects_uuid(request):
    try:
        company = json.loads(request.body)
        uid = str(company['uuid'])
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "Equal",
                "valueText": uid
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECTS UUID:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_objects_uuids(request):
    try:
        company = json.loads(request.body)
        uids = list(company['uuids'])
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        for uuid in uids:
            llm_hybrid.weaviate_client.batch.delete_objects(
                class_name=collection,
                where={
                    "path": ["uuid"],
                    "operator": "Equal",
                    "valueText": uuid
                },
            )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE OBJECTS UUIDS:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_collection(request):
    try:
        company = json.loads(request.body)
        collection = "C" + str(company['collection'])

        if llm_hybrid.collection_exists(collection) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        llm_hybrid.weaviate_client.schema.delete_class(collection)
        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW REMOVE COLLECTION:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ------------------------- Master Vectors -------------------------
@api_view(http_method_names=['POST'])
def create_master_collection(request):
    try:
        company = json.loads(request.body)

        if mv.weaviate_client.schema.exists(company['collection']) is True:
            return Response({'error': 'This collection already exists!'}, status=status.HTTP_400_BAD_REQUEST)

        class_obj = {
            "class": f"{company['collection']}",
            "description": f"collection for {company['collection']}",
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
                    "name": "filename",
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
                },
                {
                    "name": "type",  # EA, Finance data, block chains, .....
                    "dataType": ["text"],
                    "moduleConfig": {
                        "text2vec-cohere": {
                            "skip": True
                        }
                    }
                }
            ],
        }

        mv.weaviate_client.schema.create_class(class_obj)
        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_collection(request):
    try:
        company = json.loads(request.body)

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = (mv.weaviate_client.query.get(company['collection'], ['uuid', 'filename', 'content', 'type'])
                       .with_additional(["id"])
                       .do())

        res = data_object['data']['Get']
        # print(res)
        return Response({'msg': res}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_objects_filename(request):
    try:
        company = json.loads(request.body)
        filename = company['filename']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = mv.filter_by('filename', filename, collection)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_objects_type(request):
    try:
        company = json.loads(request.body)
        type_ = company['type']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = mv.filter_by('type', type_, collection)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_objects_uuid(request):
    try:
        company = json.loads(request.body)
        uid = company['uuid']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = mv.filter_by('uuid', uid, collection)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['GET'])
def get_master_object(request):
    try:
        company = json.loads(request.body)
        obj_id = company['id']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        data_object = mv.get_by_id(collection, obj_id)
        return Response({'msg': data_object}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_master_object(request):
    try:
        company = json.loads(request.body)
        obj_id = company['id']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        mv.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["id"],
                "operator": "Like",
                "valueText": obj_id
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_master_collection(request):
    try:
        company = json.loads(request.body)
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        mv.weaviate_client.schema.delete_class(collection)
        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_master_objects_uuid(request):
    try:
        company = json.loads(request.body)
        uid = company['uuid']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        mv.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["uuid"],
                "operator": "Like",
                "valueText": uid
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def remove_master_objects_file(request):
    try:
        company = json.loads(request.body)
        filename = company['filename']
        collection = company['collection']

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        mv.weaviate_client.batch.delete_objects(
            class_name=collection,
            where={
                "path": ["filename"],
                "operator": "Like",
                "valueText": filename
            },
        )

        return Response({'msg': 'Success!'}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def add_master_vectors(request):
    try:
        company = json.loads(request.body)

        if mv.weaviate_client.schema.exists(company['collection']) is False:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        documents = slice_document.chunk_corpus(company['text'])

        uid = mv.add_batch(documents, company['filename'], company['type'], company['collection'])

        return Response({'msg': str(uid)}, status=status.HTTP_200_OK)
    except Exception as e:
        print("VIEW:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(http_method_names=['POST'])
def upload_master_file(request):
    try:
        data = request.data
        documents = data.get('chunks')
        file_name = data.get('filename')
        doc_type = data.get('type')
        collection = data.get('collection')

        if not documents or not file_name or not doc_type or not collection:
            return Response({'msg': 'Invalid data'}, status=status.HTTP_400_BAD_REQUEST)

        # Add batch processing logic
        uid = mv.add_batch(documents, file_name, doc_type, collection)

        return Response({'msg': uid}, status=status.HTTP_201_CREATED)
    except Exception as e:
        print("VIEW MASTER UPLOAD FILE:")
        print(e)
        return Response({'msg': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ----------- WARNING -----------
@api_view(http_method_names=['POST'])
def destroy_all(request):
    llm_hybrid.weaviate_client.schema.delete_all()
    return Response({'msg': 'Destroyed!!!'}, status=status.HTTP_200_OK)
