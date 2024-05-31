from celery import shared_task
from .models import Company
import json
import os
import uuid
from weaviate_client import WeaviateClient

mv = WeaviateClient()

@shared_task
def add_master_vectors_task(company_data):
    try:
        company_data = json.loads(company_data)
        company = Company.objects.get(collection=company_data['collection'])

        if not company:
            return Response({'error': 'This collection does not exist!'}, status=status.HTTP_400_BAD_REQUEST)

        documents = slice_document.chunk_corpus(company_data['text'])

        uid = mv.add_batch(documents, company_data['filename'], company_data['type'], company_data['collection'])

        return Response({'msg': str(uid)}, status=status.HTTP_200_OK)
    except Exception as e:
        print("TASK:")
        print(e)
        return Response({'error': 'Something went wrong!'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
