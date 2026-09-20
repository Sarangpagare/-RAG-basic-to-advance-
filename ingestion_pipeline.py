import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):
    """load all the text files from docs directory"""
    print(f"loading documents from {docs_path}...")

    #check if the docs_path exists 
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"the directory {docs_path} does not exist. Please create it and add some text files.")

    #load all the text files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader  
    )


    documents = loader.load()
    if len(documents) == 0:
        raise FileNotFoundError(f"no text file found in {docs_path}, please addd some text files.")

    for i,doc in enumerate(documents[:2]): #show first 2 documents
        print(f"\nDocument{i+1}:")
        print(f"Source: {doc.metadata['source']}")
        print(f"Content length: {len(doc.page_content)} characters")
        print(f"Content preview: {doc.page_content[:100]}...")
        print(f"metadata: {doc.metadata}")

    return documents



def main():
    print("Main Function")

    #1. Load documents
    documents = load_documents(docs_path="docs")
    #2. Chunking the files
    #3. create embeddings and storing in vector DB
if __name__ == "__main__":
    main()
    
