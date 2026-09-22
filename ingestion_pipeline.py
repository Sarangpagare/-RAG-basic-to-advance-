import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os 
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()


# ::This is part of the ingestion pipeline, which is responsible for loading the documents from the docs directory and returning them as a list of Document objects.
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
        loader_cls=TextLoader,  
        loader_kwargs={"encoding": "utf-8"}
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

# ::Here we will do the Chunking part of the Documents  
def split_documents(documents , chunk_size=1000, chunk_overlap=0):
    '''splits document into smaller chunks with overlap '''  
    print("Splitting documents into chunks...")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:

        for i, chunk in enumerate(chunks[:5]):
            print(f"\n---chunk{i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)}character")
            print(f"Content: ")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks  

#here we will do vector embeddings and storing it into vector database
def create_vector_store(chunks, persist_directory="db/chroma_db"):
    '''create and persist ChromaDB vector store'''
    print("creating embedding and storing it in ChromaDB...")

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    #create Chroma vector store
    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents = chunks,
        embedding = embedding_model,
        persist_directory = persist_directory,
        collection_metadata = { "hnsw:space", "cosine" }
    )  
    print("--- finished creating vector store---")

    print(f"vector store craeted and saved to {persist_directory}")  
    return vectorstore 
   


def main():
    print("Main Function")

    #1. Load documents
    documents = load_documents(docs_path="docs")

    #2. Chunking the files
    chunks = split_documents(documents)

    #3. create embeddings and storing in vector DB
    vectorstore = create_vector_store(chunks)



if __name__ == "__main__":
    main()
    
