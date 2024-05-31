release: python manage.py migrate

web: gunicorn main.wsgi --workers 3 --timeout 30
