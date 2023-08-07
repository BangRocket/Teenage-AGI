import os
from dotenv import load_dotenv
# Load default environment variables (.env)
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL") or "gpt-4"

ACTIVELOOP_KEY = os.getenv("ACTIVELOOP_KEY")
ACTIVELOOP_TOKEN = os.getenv("ACTIVELOOP_TOKEN")
ACTIVELOOP_USER = os.getenv("ACTIVELOOP_USER")
AGENT_NAME = os.getenv("AGENT_NAME")