from flask import Flask, request, jsonify
from flask_cors import CORS
from together import Together
import os
import logging
from dotenv import load_dotenv
import pandas as pd
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import psycopg2
import hashlib
import time
import json
from typing import List

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

class TogetherAIEmbeddings(Embeddings):
    def __init__(self):
        self.model_name = "togethercomputer/m2-bert-80M-8k-retrieval"  # Good for retrieval tasks
        # Alternatively: "togethercomputer/m2-bert-80M-2k-retrieval" for shorter documents
        self.api_key = os.getenv('TOGETHER_API_KEY')
        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts using TogetherAI API"""
        try:
            # Batch process texts to be more efficient
            batch_size = 32  # Adjust based on API limits
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(
                    input=batch,
                    model=self.model_name
                )
                batch_embeddings = [emb.embedding for emb in response.data]
                embeddings.extend(batch_embeddings)
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query using TogetherAI API"""
        try:
            response = client.embeddings.create(
                input=[text],
                model=self.model_name
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
        
    def __del__(self):
        """Cleanup when the object is destroyed"""
        pass  # No special cleanup needed for API-based embeddings

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
                # Create document with full metadata
                doc = Document(
                    page_content=desc,
                    metadata=metadata
                )
                
                documents.append(doc)
        
        print(f"Loaded {len(documents)} documents from Excel file")
        return documents, get_file_hash(excel_path)
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

def get_collection_id() -> str:
    """Get the UUID of the current collection"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        collection_name = get_collection_name()
        
        cur.execute("""
            SELECT uuid FROM langchain_pg_collection WHERE name = %s;
        """, (collection_name,))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        if result:
            return result[0]
        else:
            raise Exception(f"Collection {collection_name} not found")
            
    except Exception as e:
        logger.error(f"Error getting collection ID: {str(e)}")
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
            logger.info("file_hashes table does not exist. Creating it...")
            # Create hash table if it doesn't exist
            cur.execute("""
                CREATE TABLE file_hashes (
                    collection_name TEXT PRIMARY KEY,
                    file_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()
            logger.info("Created file_hashes table")
            cur.close()
            conn.close()
            return None
        
        # Get stored hash
        cur.execute("""
            SELECT file_hash FROM file_hashes WHERE collection_name = %s;
        """, (collection_name,))
        result = cur.fetchone()
        
        if result is None:
            logger.info(f"No hash found for collection: {collection_name}")
        else:
            logger.info(f"Found hash for collection: {collection_name}")
            
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
        
        # Ensure file_hashes table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS file_hashes (
                collection_name TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Update or insert hash
        cur.execute("""
            INSERT INTO file_hashes (collection_name, file_hash, updated_at)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (collection_name) 
            DO UPDATE SET 
                file_hash = EXCLUDED.file_hash,
                updated_at = CURRENT_TIMESTAMP;
        """, (collection_name, file_hash))
        
        conn.commit()
        logger.info(f"Updated hash for collection: {collection_name}")
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error updating stored hash: {str(e)}")
        raise

# Global embeddings instance
embeddings = None
query_vectorstore = None

def get_embeddings():
    """Get or create the embeddings instance"""
    global embeddings
    if embeddings is None:
        logger.info("Creating new TogetherAI embeddings instance...")
        embeddings = TogetherAIEmbeddings()
    return embeddings

def get_collection_name() -> str:
    """Get the collection name from environment variable or use default"""
    collection_name = os.getenv('COLLECTION_NAME')
    if not collection_name:
        logger.info("No COLLECTION_NAME environment variable set. Using default: art_of_living_projects")
        collection_name = 'art_of_living_projects'
    else:
        logger.info(f"Using collection name from environment: {collection_name}")
    return collection_name

def get_query_vectorstore():
    """Get or create the query vector store with embeddings"""
    global query_vectorstore, embeddings
    if query_vectorstore is None:
        logger.info("Creating query vector store...")
        embeddings = get_embeddings()
        collection_name = get_collection_name()
        database_url = os.getenv('DATABASE_URL')
        query_vectorstore = PGVector(
            connection_string=database_url,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    return query_vectorstore

def verify_collection_integrity(connection_string: str, collection_name: str) -> bool:
    """Verify the integrity of a collection by checking its structure and content"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check if collection exists
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (collection_name,))
        if not cur.fetchone()[0]:
            return False
            
        # Check if collection has required columns
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = %s;
        """, (collection_name,))
        columns = [row[0] for row in cur.fetchall()]
        required_columns = {'id', 'embedding', 'document', 'metadata'}
        if not required_columns.issubset(set(columns)):
            return False
            
        # Check if collection has data
        cur.execute(f"SELECT COUNT(*) FROM {collection_name}")
        count = cur.fetchone()[0]
        if count == 0:
            return False
            
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error verifying collection integrity: {str(e)}")
        return False

def backup_collection(connection_string: str, source_collection: str, backup_collection: str):
    """Efficiently backup a collection using SQL"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Drop backup collection if it exists
        cur.execute(f"DROP TABLE IF EXISTS {backup_collection}")
        
        # Create backup by copying the entire table
        cur.execute(f"""
            CREATE TABLE {backup_collection} AS 
            SELECT * FROM {source_collection}
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error backing up collection: {str(e)}")
        return False

def init_vectorstore():
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    collection_name = get_collection_name()
    database_url = os.getenv('DATABASE_URL')
    excel_path = os.path.join(data_dir, "Data Sample for Altro AI-1.xlsx")
    
    # Check if Excel file exists
    if not os.path.exists(excel_path):
        logger.error(f"Excel file not found: {excel_path}")
        # If collection exists, use it; otherwise raise error
        if collection_exists(database_url, collection_name):
            logger.warning("Using existing collection as Excel file is missing")
            return get_query_vectorstore()
        raise FileNotFoundError(f"Excel file not found and no existing collection")
    
    exists = collection_exists(database_url, collection_name)
    stored_hash = get_stored_hash(database_url, collection_name)
    current_hash = get_file_hash(excel_path)
    
    try:
        # If collection exists and hashes match, verify integrity
        if exists and stored_hash == current_hash:
            if verify_collection_integrity(database_url, collection_name):
                logger.info(f"Collection {collection_name} exists and is up to date")
                return get_query_vectorstore()
            else:
                logger.warning("Collection integrity check failed, will attempt repair")
        
        # Create backup if updating existing collection
        if exists:
            backup_name = f"{collection_name}_backup_{int(time.time())}"
            logger.info(f"Creating backup: {backup_name}")
            if backup_collection(database_url, collection_name, backup_name):
                # Clean up old backups (keep last 3)
                cleanup_old_backups(database_url, collection_name, keep_last=3)
            else:
                logger.error("Backup creation failed")
                raise Exception("Failed to create backup before update")
        
        # Load and update documents
        logger.info("Loading documents for update...")
        documents, _ = load_documents()
        
        # Create or update vector store
        vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=get_embeddings(),
            collection_name=collection_name,
            connection_string=database_url,
            pre_delete_collection=True,
            collection_metadata={"name": collection_name}  # Add collection metadata
        )
        
        # Verify metadata storage
        try:
            test_doc = vectorstore.similarity_search("test", k=1)[0]
            print(f"Test document metadata after storage: {test_doc.metadata}")
        except Exception as e:
            print(f"Error verifying metadata: {str(e)}")
        
        # Update hash only after successful update
        update_stored_hash(database_url, collection_name, current_hash)
        
        # Update global query store
        global query_vectorstore
        query_vectorstore = vectorstore
        
        logger.info("Vector store update complete!")
        return vectorstore
        
    except Exception as e:
        logger.error(f"Error during vector store initialization: {str(e)}")
        if exists:
            logger.info("Attempting to restore from latest backup...")
            latest_backup = get_latest_backup(database_url, collection_name)
            if latest_backup and restore_from_backup(database_url, latest_backup, collection_name):
                logger.info("Successfully restored from backup")
                return get_query_vectorstore()
        raise

def cleanup_old_backups(database_url: str, collection_prefix: str, keep_last: int = 3):
    """Clean up old backup collections, keeping the specified number of most recent backups"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get all backup collections for this prefix
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE %s 
            ORDER BY table_name DESC
        """, (f"{collection_prefix}_backup_%",))
        
        backups = [row[0] for row in cur.fetchall()]
        
        # Keep only the specified number of most recent backups
        for backup in backups[keep_last:]:
            cur.execute(f"DROP TABLE IF EXISTS {backup}")
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        logger.error(f"Error cleaning up old backups: {str(e)}")

def get_latest_backup(database_url: str, collection_name: str) -> str:
    """Get the name of the most recent backup collection"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE %s 
            ORDER BY table_name DESC 
            LIMIT 1
        """, (f"{collection_name}_backup_%",))
        
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        return result[0] if result else None
        
    except Exception as e:
        logger.error(f"Error getting latest backup: {str(e)}")
        return None

def restore_from_backup(database_url: str, backup_name: str, target_name: str) -> bool:
    """Restore a collection from a backup"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Drop target if it exists
        cur.execute(f"DROP TABLE IF EXISTS {target_name}")
        
        # Restore from backup
        cur.execute(f"""
            CREATE TABLE {target_name} AS 
            SELECT * FROM {backup_name}
        """)
        
        conn.commit()
        cur.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"Error restoring from backup: {str(e)}")
        return False

def cleanup_and_init_db():
    """Clean up existing collections and initialize a fresh database"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        collection_name = get_collection_name()
        
        # Drop existing collection and related tables with CASCADE
        cur.execute(f"DROP TABLE IF EXISTS {collection_name} CASCADE")
        cur.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE")
        cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE")
        cur.execute("DROP TABLE IF EXISTS file_hashes CASCADE")
        
        conn.commit()
        cur.close()
        conn.close()
        
        logger.info("Cleaned up existing database tables")
        
        # Initialize fresh vector store
        return init_vectorstore()
        
    except Exception as e:
        logger.error(f"Error during database cleanup and initialization: {str(e)}")
        raise

# Initialize vector store during app startup
logger.info("Starting application initialization...")
try:
    # Choose one of these options:
    # Option 1: Clean up and reinitialize the database (for development only)
    # vectorstore = cleanup_and_init_db()
    
    # Option 2: Normal initialization (preserves existing data) - Use this in production
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

@app.route("/ask", methods=["POST", "OPTIONS"])
def ask():
    # Handle preflight requests
    if request.method == "OPTIONS":
        return "", 204

    try:
        print("Received request:", request.get_json())
        data = request.get_json()
        if not data or "query" not in data:
            return jsonify({"error": "Query is required"}), 400

        user_input = data["query"]
        if not user_input:
            return jsonify({"error": "Query cannot be empty"}), 400

        print("Processing query:", user_input)
        
        # Get relevant projects from vector store
        project_info = ""
        if vectorstore:
            try:
                relevant_projects = vectorstore.similarity_search(user_input, k=3)
                # Format project information with detailed metadata
                project_info = "\n\nRelevant Projects:\n"
                for i, project in enumerate(relevant_projects, 1):
                    project_info += f"\n{i}. {project.metadata.get('title', 'Untitled Project')}\n"
                    project_info += f"   Description: {project.page_content}\n"
                    if project.metadata.get('contact_email'):
                        project_info += f"   Contact: {project.metadata['contact_email']}\n"
                    project_info += f"   Link: {project.metadata.get('link', 'N/A')}\n"
            except Exception as e:
                print(f"Error accessing vector store: {str(e)}")
                print("Full error details:", e.__dict__)
        else:
            print("Warning: Vector store is not initialized!")
        
        # Create enhanced prompt with project information
        prompt = f"""
You are an AI assistant for the **Art of Living**, dedicated to spreading peace, well-being, and service.

### INSTRUCTIONS
1️⃣ Recommend specific projects from the database.  
2️⃣ Match based on location, interests, and budget.  
3️⃣ For **donations**, only show projects within budget.  
4️⃣ If no exact match, suggest the closest options with a reason.  
5️⃣ For **volunteering**, match based on location and skills.  
6️⃣ Never invent projects.

### Relevant Projects:
{project_info}

User Query: {user_input}
"""

        print("\n=== Sending to Together AI ===")
        print("Prompt:", prompt)
        
        response = client.chat.completions.create(
            model=os.getenv('MODEL_NAME'),
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that helps users find relevant projects based on their queries. You have access to project data and can provide detailed information about projects that match the user's interests."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1024,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
        )

        full_response = response.choices[0].message.content
        print("\n=== Together AI Response ===")
        print("Response:", full_response)
        
        return jsonify({
            "response": full_response,
            "environment": "production" if os.getenv('FLASK_ENV') == 'production' else "development"
        })

    except Exception as e:
        print("\n=== Error Occurred ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Full error details:", e.__dict__)
        return jsonify({"error": str(e)}), 500

@app.route('/test-db')
def test_db():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get collection details
        cur.execute("""
            SELECT 
                name,
                uuid,
                cmetadata
            FROM langchain_pg_collection
            ORDER BY name;
        """)
        collections = cur.fetchall()
        
        # Get embedding counts for each collection
        collection_data = []
        for collection in collections:
            name, uuid, metadata = collection
            
            # Get embedding count
            cur.execute(f"""
                SELECT COUNT(*) 
                FROM langchain_pg_embedding 
                WHERE collection_id = %s;
            """, (uuid,))
            embedding_count = cur.fetchone()[0]
            
            # Get file hash
            cur.execute("""
                SELECT file_hash 
                FROM file_hashes 
                WHERE collection_name = %s;
            """, (name,))
            hash_result = cur.fetchone()
            file_hash = hash_result[0] if hash_result else None
            
            collection_data.append({
                "name": name,
                "uuid": str(uuid),
                "metadata": metadata,
                "embedding_count": embedding_count,
                "file_hash": file_hash
            })
        
        # Get sample documents from the current collection
        collection_name = get_collection_name()
        cur.execute("""
            SELECT 
                document,
                cmetadata
            FROM langchain_pg_embedding
            WHERE collection_id = (
                SELECT uuid 
                FROM langchain_pg_collection 
                WHERE name = %s
            )
            LIMIT 5;
        """, (collection_name,))
        sample_documents = cur.fetchall()
        
        # Format sample documents with both content and metadata
        formatted_documents = []
        for doc, metadata in sample_documents:
            formatted_documents.append({
                "document": {
                    "content": doc,
                    "metadata": metadata
                }
            })
        
        cur.close()
        conn.close()
        
        return jsonify({
            "status": "success",
            "database_connection": "ok",
            "collections": collection_data,
            "current_collection": {
                "name": collection_name,
                "sample_documents": formatted_documents
            }
        })
        
    except Exception as e:
        logger.error(f"Database test failed: {str(e)}")
        return jsonify({
            "status": "error",
            "database_connection": "failed",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Get port dynamically from Render's environment or default to 8000
    port = int(os.getenv('PORT', 8000))
    app.run(host='0.0.0.0', port=port)