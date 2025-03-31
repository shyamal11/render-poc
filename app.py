from flask import Flask, request, jsonify
from flask_cors import CORS
from together import Together
import os
import logging
from dotenv import load_dotenv
import pandas as pd
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
import torch
from transformers import AutoModel, AutoTokenizer
from langchain_core.embeddings import Embeddings
import psycopg2
import hashlib

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize Together AI client
client = Together()
client.api_key = os.getenv('TOGETHER_API_KEY')

# Configure CORS
CORS(app, resources={r"/*": {"origins": "*"}})

def get_db_connection():
    """Get a database connection"""
    return psycopg2.connect(os.getenv('DATABASE_URL'))

def get_file_hash(file_path):
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

class NomicEmbeddings(Embeddings):
    def __init__(self):
        self.model_name = "nomic-ai/nomic-embed-text-v1.5"
        self.max_length = 8192
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Load model with memory optimizations
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                device_map="auto",  # Automatically handle model placement
                torch_dtype=torch.float16,  # Use half precision
                low_cpu_mem_usage=True  # Optimize memory usage
            )
            
            # Move model to appropriate device
            self.model = self.model.to(self.device)
            
            # Enable evaluation mode
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            texts = [str(t) for t in texts]
            encoded_input = self.tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                max_length=self.max_length, 
                return_tensors='pt'
            )
            
            # Move inputs to the same device as the model
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded_input)
            
            # Get embeddings and move to CPU
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return embeddings.tolist()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("Out of memory error occurred. Clearing cache and retrying...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise
            raise

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
        
    def __del__(self):
        """Cleanup when the object is destroyed"""
        try:
            if hasattr(self, 'model'):
                del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")

def load_documents():
    """
    Load data from Excel file and convert to LangChain documents
    """
    try:
        # Get the data directory path
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        excel_path = os.path.join(data_dir, "Data Sample for Altro AI-1.xlsx")
        
        # Read Excel file
        xls = pd.ExcelFile(excel_path)
        data = pd.read_excel(xls, "REAL and Mocked up Data for POC").fillna("")
        
        documents = []
        for _, row in data.iterrows():
            def safe(field):
                val = row.get(field, "")
                return "" if pd.isna(val) else str(val)

            metadata = {
                "title": safe("Project title"),
                "link": safe("Links/Sources"),
                "locations": safe("Project Locations"),
                "target_group": safe("Target group"),
                "persona": safe("Persona"),
                "contact_person": safe("Contact Person"),
                "contact_title": safe("Contact Person Title"),
                "contact_email": safe("Contact Person email"),
                "volunteer_needs": safe("Volunteer Needs"),
                "volunteer_need_by": safe("Volunteer Need by (mm/dd/yyyy)"),
                "tenure": safe("Tenure"),
                "responsibilities": safe("Responsbilities"),
                "donation_needs": safe("Donation Needs"),
                "donation_amount": safe("Donation Amount ($)"),
                "donation_need_by": safe("Donation Need by"),
            }

            desc = row.get("Generated Description", "")
            if desc and isinstance(desc, str):
                documents.append(Document(page_content=desc, metadata=metadata))
        
        print(f"Loaded {len(documents)} documents from Excel file")
        return documents, get_file_hash(excel_path)
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def collection_exists(connection_string: str, collection_name: str) -> bool:
    """Check if the collection exists in the database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (collection_name,))
        exists = cur.fetchone()[0]
        cur.close()
        conn.close()
        return exists
    except Exception as e:
        logger.error(f"Error checking collection existence: {str(e)}")
        return False

def get_stored_hash(connection_string: str, collection_name: str) -> str:
    """Get the stored hash of the Excel file from the database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if hash table exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'file_hashes'
            );
        """)
        table_exists = cur.fetchone()[0]
        
        if not table_exists:
            # Create hash table if it doesn't exist
            cur.execute("""
                CREATE TABLE file_hashes (
                    collection_name TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL
                );
            """)
            conn.commit()
            cur.close()
            conn.close()
            return None
        
        # Get stored hash
        cur.execute("""
            SELECT file_hash FROM file_hashes WHERE collection_name = %s;
        """, (collection_name,))
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result[0] if result else None
    except Exception as e:
        logger.error(f"Error getting stored hash: {str(e)}")
        return None

def update_stored_hash(connection_string: str, collection_name: str, file_hash: str):
    """Update the stored hash of the Excel file in the database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO file_hashes (collection_name, file_hash)
            VALUES (%s, %s)
            ON CONFLICT (collection_name) 
            DO UPDATE SET file_hash = EXCLUDED.file_hash;
        """, (collection_name, file_hash))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error updating stored hash: {str(e)}")

# Global embeddings instance
embeddings = None
query_vectorstore = None

def get_embeddings():
    """Get or create the embeddings instance"""
    global embeddings
    if embeddings is None:
        logger.info("Creating new embeddings instance...")
        embeddings = NomicEmbeddings()
    return embeddings

def get_query_vectorstore():
    """Get or create the query vector store with embeddings"""
    global query_vectorstore, embeddings
    if query_vectorstore is None:
        logger.info("Creating query vector store...")
        embeddings = get_embeddings()
        collection_name = os.getenv('COLLECTION_NAME', 'langchain_pg_collection')
        database_url = os.getenv('DATABASE_URL')
        query_vectorstore = PGVector(
            connection_string=database_url,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    return query_vectorstore

def init_vectorstore():
    # Get the data directory path
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Get collection name and database URL
    collection_name = os.getenv('COLLECTION_NAME', 'langchain_pg_collection')
    database_url = os.getenv('DATABASE_URL')
    
    # First check if collection exists and get stored hash
    exists = collection_exists(database_url, collection_name)
    stored_hash = get_stored_hash(database_url, collection_name)
    
    # If collection exists, check if we need to update
    if exists:
        # Get current file hash without loading the data
        excel_path = os.path.join(data_dir, "Data Sample for Altro AI-1.xlsx")
        current_hash = get_file_hash(excel_path)
        
        # If hashes match, just return the query vector store
        if stored_hash == current_hash:
            logger.info(f"Collection {collection_name} exists and data is up to date")
            return get_query_vectorstore()
    
    # Only create embeddings if we need to update the collection
    logger.info(f"Loading documents and updating collection {collection_name}...")
    embeddings = get_embeddings()  # Create embeddings only when needed
    documents, current_hash = load_documents()
    
    try:
        # Create backup collection name
        backup_collection = f"{collection_name}_backup"
        
        # If collection exists, create a backup
        if exists:
            logger.info(f"Creating backup of collection {collection_name}...")
            backup_vectorstore = PGVector(
                connection_string=database_url,
                collection_name=backup_collection,
                embedding_function=embeddings
            )
            # Copy all documents to backup
            for doc in documents:
                backup_vectorstore.add_documents([doc])
        
        # Create or update vector store with embeddings
        vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=collection_name,
            connection_string=database_url,
            pre_delete_collection=True  # Delete existing collection to update with new data
        )
        
        # Update stored hash
        update_stored_hash(database_url, collection_name, current_hash)
        
        # Update query vector store
        global query_vectorstore
        query_vectorstore = vectorstore
        
        logger.info("Vector store initialization complete!")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error updating collection: {str(e)}")
        # If we have a backup, restore it
        if exists:
            logger.info("Restoring from backup...")
            backup_vectorstore = PGVector(
                connection_string=database_url,
                collection_name=backup_collection,
                embedding_function=embeddings
            )
            # Copy all documents back
            for doc in documents:
                backup_vectorstore.add_documents([doc])
            return backup_vectorstore
        raise

# Initialize vector store during app startup
logger.info("Starting application initialization...")
try:
    vectorstore = init_vectorstore()
    logger.info("Vector store initialization complete!")
except Exception as e:
    logger.error(f"Error initializing vector store: {str(e)}")
    vectorstore = None

@app.route('/')
def home():
    return jsonify({
        "message": "Art of Living Chatbot API is running",
        "environment": "development"
    })

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.json.get('query', '')
    if not user_input:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Get response from Together AI
        response = client.chat.completions.create(
            model=os.getenv('MODEL_NAME', 'meta-llama/Llama-3.3-70B-Instruct-Turbo'),
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that helps users find relevant projects based on their queries. You have access to project data and can provide detailed information about projects that match the user's interests."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=1024,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
        )
        
        # Get relevant projects from vector store
        if vectorstore:
            try:
                relevant_projects = vectorstore.similarity_search(user_input, k=3)
                # Format project information
                project_info = "\n\nRelevant Projects:\n"
                for i, project in enumerate(relevant_projects, 1):
                    project_info += f"\n{i}. {project.metadata.get('title', 'Untitled Project')}\n"
                    project_info += f"   Description: {project.page_content}\n"
                    if project.metadata.get('contact_email'):
                        project_info += f"   Contact: {project.metadata['contact_email']}\n"
                    project_info += f"   Link: {project.metadata.get('link', 'N/A')}\n"
                
                # Add project information to the response
                response.choices[0].message.content += project_info
            except Exception as e:
                logger.error(f"Error getting relevant projects: {str(e)}")
        
        return jsonify({
            "response": response.choices[0].message.content
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/test-db', methods=['GET'])
def test_db():
    """Test database connection and collection status"""
    try:
        database_url = os.getenv('DATABASE_URL')
        
        # Get all collections (tables) in the database
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE';
        """)
        collections = [row[0] for row in cur.fetchall()]
        cur.close()
        conn.close()
        
        # Get collection sizes
        collection_sizes = {}
        for collection in collections:
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute(f"SELECT COUNT(*) FROM {collection}")
                collection_sizes[collection] = cur.fetchone()[0]
                cur.close()
                conn.close()
            except Exception as e:
                logger.error(f"Error getting size for collection {collection}: {str(e)}")
                collection_sizes[collection] = "error"
        
        return jsonify({
            "status": "success",
            "database_connection": "ok",
            "available_collections": collections,
            "collection_sizes": collection_sizes
        })
        
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Get port dynamically from Render's environment or default to 8000
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port)