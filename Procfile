release: python manage.py migrate

web: gunicorn main.wsgi
worker: celery -A main worker --loglevel=info --queues=master_vectors
