release: python manage.py migrate

web: gunicorn main.wsgi --timeout 120 --workers 3 --threads 2
