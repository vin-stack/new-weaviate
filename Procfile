release: python manage.py migrate

web: gunicorn main.wsgi --workers 3 --threads 2
worker: celery -A main worker --loglevel=info --queues=master_vectors
