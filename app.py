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

class NomicEmbeddings(Embeddings):
    def __init__(self):
        self.model_name = "nomic-ai/nomic-embed-text-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.max_length = 8192

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        texts = [str(t) for t in texts]
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors='pt'
        )
        with torch.no_grad():
            outputs = self.model(**encoded_input)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

def load_documents():
    """
    Load data from Excel file and convert to LangChain documents
    """
    try:
        # Get the data directory path
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        excel_path = os.path.join(data_dir, "Data Sample for Altro AI.xlsx")
        
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
        return documents
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Initialize vector store
def init_vectorstore():
    # Get the data directory path
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Initialize embeddings
    embeddings = NomicEmbeddings()
    
    # Initialize or get existing vector store
    vectorstore = PGVector.from_documents(
        documents=load_documents(),
        embedding=embeddings,
        collection_name=os.getenv('COLLECTION_NAME', 'art_of_living_projects'),
        connection_string=os.getenv('DATABASE_URL'),
        pre_delete_collection=True
    )
    
    print("Vector store initialization complete!")
    return vectorstore

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
        "environment": "production" if os.getenv('FLASK_ENV') == 'production' else "development"
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
        projects_info = ""
        if vectorstore:
            try:
                relevant_projects = vectorstore.similarity_search(user_input, k=3)
                projects_info = "\n\n".join([
                    f"**{doc.metadata['title']}**\n- {doc.page_content}" for doc in relevant_projects
                ])
            except Exception as e:
                print(f"Error accessing vector store: {str(e)}")
        
        # Create prompt with project information
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
{projects_info}

User Query: {user_input}
"""

        print("Sending request to Together AI")
        stream = client.chat.completions.create(
            model=os.getenv('MODEL_NAME', 'togethercomputer/llama-2-70b-chat'),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )

        full_response = stream.choices[0].message.content
        print("Received response from Together AI")
        
        return jsonify({
            "response": full_response,
            "environment": "production" if os.getenv('FLASK_ENV') == 'production' else "development"
        })

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # Get port dynamically from Render's environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Application will run on port: {port}")
    app.run(host='0.0.0.0', port=port)
