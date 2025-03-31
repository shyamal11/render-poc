from langchain_community.vectorstores import PGVector
from app.embeddings import NomicEmbeddings
from app.data_loader import load_documents
from app.config import COLLECTION_NAME, DATABASE_URL

# Initialize vectorstore as a global variable
vectorstore = None

def init_vectorstore():
    global vectorstore
    if vectorstore is None:
        print("First time initialization: Loading documents and creating vector store...")
        documents = load_documents()
        embeddings = NomicEmbeddings()
        vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_string=DATABASE_URL,
            pre_delete_collection=True
        )
        print("Vector store initialization complete!")
    return vectorstore
