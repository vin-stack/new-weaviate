from . import consumer
from django.urls import re_path

websocket_urlpatterns = [
   re_path(r'socket-server/production/', consumer.ChatConsumer.as_asgi()),
]
