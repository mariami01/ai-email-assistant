import os
from dotenv import load_dotenv

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from functions import draft_email


# get and load the OpenAI API key
load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

# load documents
client_loader = CSVLoader(file_path="Data/clients.csv")
founder_loader = CSVLoader(file_path="Data/founders.csv")
roles_loader = CSVLoader(file_path="Data/roles.csv")



# create index using the loaded documents 
index_creator = VectorstoreIndexCreator()
docsearch = index_creator.from_loaders([client_loader, founder_loader, roles_loader])

# create question answering chain using index 
chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever = docsearch.vectorstore.as_retriever(), input_key="question")

# custom email prompt 


# pass a query to a chain
query = input("Prompt: ")
# response = chain({"question": query})
print(draft_email(query))