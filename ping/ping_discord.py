import os
from dotenv import load_dotenv
import discord
import asyncio
from discord.ext import commands
import logging

load_dotenv()

logging.getLogger('discord.gateway').setLevel(logging.INFO)

intents = discord.Intents.default()
intents.message_content = True

client = commands.Bot(command_prefix='!', intents=intents)


@client.event
async def on_ready():
    print(f'Logged in as {client.user}')

    channel = client.get_channel(int (os.getenv('DISCORD_CHANNEL_ID')))
    if channel:
        await channel.send("Hello from your bot!")
    else:
        print("Channel not found.")

    await client.close()  # Optional: bot exits after sending one message

client.run(os.getenv('DISCORD_BOT_TOKEN'))