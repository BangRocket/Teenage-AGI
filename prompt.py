import os
import openai
import yaml
from envkeys import OPENAI_MODEL

def generate(prompt):
	completion = openai.ChatCompletion.create(
	model=OPENAI_MODEL,
	messages=[
		{"role": "system", "content": "You are an intelligent agent with thoughts and memories. You have a memory which stores your past thoughts and actions and also how other users have interacted with you."},
		{"role": "system", "content": "Keep your thoughts relatively simple and concise"},
		{"role": "user", "content": prompt},
		]
	)
	return completion.choices[0].message["content"]

# Prompt Initialization
with open('prompts.yaml', 'r') as f:
	prompts = yaml.load(f, Loader=yaml.FullLoader)