release: python manage.py migrate

web: gunicorn main.wsgi --workers 3 --threads 2
worker: celery worker --app=main.app
