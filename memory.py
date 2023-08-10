import openai
import sys
import yaml
import nltk
import numpy as np
from langchain.text_splitter import NLTKTextSplitter
from deeplake.core.vectorstore import VectorStore
from envkeys import OPENAI_MODEL, OPENAI_API_KEY, ACTIVELOOP_KEY, ACTIVELOOP_TOKEN, ACTIVELOOP_USER, AGENT_NAME
from prompt import prompts, generate

# Thought types, used in Pinecone Namespace
THOUGHTS = "Thoughts"
QUERIES = "Queries"
INFORMATION = "Information"
ACTIONS = "Actions"

# Download NLTK for Reading
nltk.download('punkt')

# Initialize Text Splitter
text_splitter = NLTKTextSplitter(chunk_size=2500)

# Counter Initialization
with open('memory_count.yaml', 'r') as f:
	memory_counter = yaml.load(f, Loader=yaml.FullLoader)

# internalThoughtPrompt = prompts['internal_thought']
# externalThoughtPrompt = prompts['external_thought']
# internalMemoryPrompt = prompts['internal_thought_memory']
# externalMemoryPrompt = prompts['external_thought_memory']

def get_ada_embedding(text):
		text = text.replace("\n", " ")
		return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
			"data"
		][0]["embedding"]

class Memory():
	def __init__(self, table_name=None) -> None:
		self.table_name = table_name
		self.thought_id_count = int(memory_counter['count'])
		
	# Keep Remebering!
	# def __del__(self) -> None:
	#     with open('memory_count.yaml', 'w') as f:
	#         yaml.dump({'count': str(self.thought_id_count)}, f)
	

	def createIndex(self, table_name=None):
		# Create Pinecone index
		if(table_name):
			self.table_name = table_name

		if(self.table_name == None):
			return

		print(f"attempting to create/load from hub://{ACTIVELOOP_USER}/{AGENT_NAME}")

		self.memory = VectorStore(
			path = f"hub://{ACTIVELOOP_USER}/{AGENT_NAME}",
			tensor_params = [
					{'name': 'embedding', 'htype': 'embedding'},
					{'name': 'id', 'htype': 'text'},
					{'name': 'type', 'htype': 'text'},
					{'name': 'metadata', 'htype': 'text'},				 
					{'name': 'text', 'htype': 'text'}
					],
			overwrite=False,
			num_workers=5
		)
		# dimension = 1536``
		# metric = "cosine"
		# pod_type = "p1"
		# if self.table_name not in pinecone.list_indexes():
		# 	pinecone.create_index(
		# 		self.table_name, dimension=dimension, metric=metric, pod_type=pod_type
		# 	)

		# embedding, id, metadata, text
		# # Give memory
		# self.memory = pinecone.Index(self.table_name)

	
	# Adds new Memory to agent, types are: THOUGHTS, ACTIONS, QUERIES, INFORMATION
	def updateMemory(self, new_thought, thought_type):
		with open('memory_count.yaml', 'w') as f:
			yaml.dump({'count': str(self.thought_id_count)}, f)

		if thought_type==INFORMATION:
			source = "This is information fed to you by the user:\n" + new_thought
		elif thought_type==QUERIES:
			source = "The user has said to you before:\n" + new_thought
		elif thought_type==THOUGHTS:
			# Not needed since already in prompts.yaml
			source = "You have previously thought:\n" + new_thought
			pass
		elif thought_type==ACTIONS:
			# Not needed since already in prompts.yaml as external thought memory
			source = "external thought memory:\n" + new_thought
			pass

		vector = get_ada_embedding(new_thought)

		vector=[vector]
		id=[f"thought-{self.thought_id_count}"]
		type=[thought_type]
		metadata=[source]
		text=[new_thought]

		upsert_response = self.memory.add(
			embedding=vector,
			id=id, 
			type=type,
			metadata=metadata,
			text=text
			)
		self.thought_id_count += 1

	# Agent thinks about given query based on top k related memories. Internal thought is passed to external thought
	def internalThought(self, query) -> str:
		query_embedding = get_ada_embedding(query)
		query_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=QUERIES)
		thought_results = self.memory.query(query_embedding, top_k=2, include_metadata=True, namespace=THOUGHTS)
		results = query_results.matches + thought_results.matches
		sorted_results = sorted(results, key=lambda x: x.score, reverse=True)
		top_matches = "\n\n".join([(str(item.metadata["thought_string"])) for item in sorted_results])
		#print(top_matches)
		
		internalThoughtPrompt = prompts['internal_thought']
		internalThoughtPrompt = internalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{last_message}", self.last_message)
		print("------------INTERNAL THOUGHT PROMPT------------")
		print(internalThoughtPrompt)
		internal_thought = generate(internalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here
		
		# Debugging purposes
		#print(internal_thought)

		internalMemoryPrompt = prompts['internal_thought_memory']
		internalMemoryPrompt = internalMemoryPrompt.replace("{query}", query).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
		self.updateMemory(internalMemoryPrompt, THOUGHTS)
		return internal_thought, top_matches

	def action(self, query) -> str:
		internal_thought, top_matches = self.internalThought(query)
		
		externalThoughtPrompt = prompts['external_thought']
		externalThoughtPrompt = externalThoughtPrompt.replace("{query}", query).replace("{top_matches}", top_matches).replace("{internal_thought}", internal_thought).replace("{last_message}", self.last_message)
		print("------------EXTERNAL THOUGHT PROMPT------------")
		print(externalThoughtPrompt)
		external_thought = generate(externalThoughtPrompt) # OPENAI CALL: top_matches and query text is used here

		externalMemoryPrompt = prompts['external_thought_memory']
		externalMemoryPrompt = externalMemoryPrompt.replace("{query}", query).replace("{external_thought}", external_thought)
		self.updateMemory(externalMemoryPrompt, THOUGHTS)
		request_memory = prompts["request_memory"]
		self.updateMemory(request_memory.replace("{query}", query), QUERIES)
		self.last_message = query
		return external_thought

	# Make agent think some information
	def think(self, text) -> str:
		self.updateMemory(text, THOUGHTS)

