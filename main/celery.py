import os
from celery import Celery
from .tasks import add_master_vectors_task # type: ignore

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'main.settings')

app = Celery('main')
app.config_from_object('django.conf:settings', namespace='CELERY')

# Set the broker URL to use Redis
app.conf.broker_url = os.environ.get('REDIS_URL')

app.autodiscover_tasks()
# add the tasks module explicitly
app.register_tasks(add_master_vectors_task)
