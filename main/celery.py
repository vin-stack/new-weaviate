import os
from celery import Celery

app = Celery('main')
app.conf.broker_url = os.environ.get('REDIS_URL')
app.conf.result_backend = os.environ.get('REDIS_URL')
app.conf.task_routes = {
    'main.tasks.add_master_vectors_task': {'queue': 'long_tasks'},
}
