release: python manage.py migrate

web: gunicorn main.wsgi --timeout 240 --workers 3 --threads 2
