from celery import shared_task
from .master_vectors.MV import MasterVectors
mv = MasterVectors()

@task()
def add_master_vectors_task(company):
    """
    A Celery task that adds master vectors to the Weaviate database
    """
    try:
        if mv.weaviate_client.schema.exists(company['collection']) is False:
            raise Exception('This collection does not exist!')

        documents = mv.slice_document.chunk_corpus(company['text'])

        uid = mv.add_batch(documents, company['filename'], company['type'], company['collection'])

        return {'msg': str(uid)}
    except Exception as e:
        print("TASK:")
        print(e)
        raise
