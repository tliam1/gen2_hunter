# discord_bot.py
from datetime import datetime
import io
import json
import cv2
import discord
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))
bot_loop = None
discord_client = None

class DiscordBot(discord.Client):
    def __init__(self, message_queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.message_queue = message_queue
        self.bg_task = None

    async def on_ready(self):
        print(f'Logged in as {self.user}')
        self.channel = self.get_channel(CHANNEL_ID)
        with open("my_data.json", 'r') as file:
            data = json.load(file)

            today = datetime.now().strftime('%Y-%m-%d')
            if today not in data:
                data[today] = 0
            data[today] += 1
        await self.channel.send(f"# 🤖 Bot is now online and hunting shinies!\n\n```md\nEncounters Today: {data[today]}\nDate of last Found Shiny: {data["last_found_shiny"] if data["last_found_shiny"] is not None else "None Recorded"}\nLifetime Encounters:{data["encounters"]}```")
        self.bg_task = self.loop.create_task(self.message_sender())
        

    async def message_sender(self):
        while True:
            msg = await self.message_queue.get()

            if isinstance(msg, tuple):  # (text, file_data)
                text, file_data = msg

                # If it's a discord.File object, send it directly
                if isinstance(file_data, discord.File):
                    await self.channel.send(content=text, file=file_data)
                else:
                    await self.channel.send(content=f"{text}\n⚠️ Invalid file format.")
            else:
                await self.channel.send(content=msg)

    async def shutdown(self):
        with open("my_data.json", 'r') as file:
            data = json.load(file)
            if hasattr(self, 'channel') and self.channel is not None:
                await self.channel.send(f"# ⚠️ Bot is shutting down! \n## Total sesssion encounters: {data["session_encounters"]}")
        await self.close()

# Shared queue for sending messages
message_queue = asyncio.Queue()

def run_bot():
    global bot_loop, discord_client
    loop = asyncio.new_event_loop()
    bot_loop = loop
    asyncio.set_event_loop(loop)

    intents = discord.Intents.default()
    discord_client = DiscordBot(message_queue, intents=intents)
    loop.run_until_complete(discord_client.start(DISCORD_TOKEN))

def image_to_discord_file(cv2_img, filename="shiny.png"):
    # Encode image to PNG format in memory
    success, encoded_image = cv2.imencode('.png', cv2_img)
    if not success:
        raise ValueError("Could not encode image")

    # Convert to BytesIO and wrap in discord.File
    image_bytes = io.BytesIO(encoded_image.tobytes())
    image_bytes.seek(0)  # Reset stream position

    return discord.File(fp=image_bytes, filename=filename)

def get_bot_loop():
    return bot_loop

def get_client():
    return discord_client