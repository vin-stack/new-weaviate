release: python manage.py migrate
web: gunicorn main.wsgi --threads=2
web: daphne -b 0.0.0.0 -p $PORT main.asgi:application 

