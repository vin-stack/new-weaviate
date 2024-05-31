release: python manage.py migrate

web: gunicorn myapp.wsgi --workers 3 --timeout 30
