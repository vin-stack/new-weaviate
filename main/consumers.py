
import json
from channels.generic.websocket import AsyncWebsocketConsumer
import asyncio

class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Simulate response generation
        for i in range(5):
            await self.send(text_data=json.dumps({
                'message': f'Part {i+1}: {message}'
            }))
            await asyncio.sleep(1)  # Simulate delay
