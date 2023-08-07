import openai
import sys
import nltk
from langchain.text_splitter import NLTKTextSplitter
from envkeys import OPENAI_MODEL, OPENAI_API_KEY, ACTIVELOOP_KEY, ACTIVELOOP_TOKEN, ACTIVELOOP_USER, AGENT_NAME
from memory import Memory, INFORMATION
# Download NLTK for Reading
nltk.download('punkt', download_dir='./nltk')

# Initialize Text Splitter
text_splitter = NLTKTextSplitter(chunk_size=2500)

# Top matches length
k_n = 3

# initialize openAI
openai.api_key = OPENAI_API_KEY # you can just copy and paste your key here if you want

def get_ada_embedding(text):
		text = text.replace("\n", " ")
		return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
			"data"
		][0]["embedding"]

def read_txtFile(file_path):
	with open(file_path, 'r', encoding='utf-8') as file:
		text = file.read()
	return text

class Agent():
	def __init__(self, table_name=None) -> None:
		self.table_name = table_name
		self.memory = Memory(table_name)
		self.last_message = ""

	# Make agent read some information
	def read(self, text) -> str:
		texts = text_splitter.split_text(text)
		for t in texts:
			self.memory.updateMemory(t,INFORMATION)

		
	# Make agent read a document
	def readDoc(self, text) -> str:
		texts = text_splitter.split_text(read_txtFile(text))
		vectors = []
		for t in texts:
			self.memory.updateMemory(t,INFORMATION)
