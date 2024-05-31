release: python manage.py migrate

web: gunicorn main.wsgi --timeout 250 --workers 3 --threads 2
