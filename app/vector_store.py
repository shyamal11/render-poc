from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from app.config import COLLECTION_NAME
from app.data_loader import load_data
import os

# Initialize the vector store
def init_vectorstore():
    # Get the data directory path
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Initialize or get existing vector store
    vectorstore = Chroma(
        persist_directory=os.path.join(data_dir, 'chroma'),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )
    
    # Load data if the collection is empty
    if vectorstore._collection.count() == 0:
        print("Loading data into vector store...")
        documents = load_data()
        vectorstore.add_documents(documents)
        vectorstore.persist()
        print("Data loaded successfully!")
    
    return vectorstore
