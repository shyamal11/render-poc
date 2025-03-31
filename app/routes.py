from flask import Blueprint, request, jsonify
from together import Together
from config import MODEL_NAME, TOGETHER_API_KEY
from vector_store import init_vectorstore
import os
import traceback

routes = Blueprint("routes", __name__)
client = Together()  # Initialize without arguments
client.api_key = TOGETHER_API_KEY  # Set API key separately

@routes.route("/")
def home():
    return jsonify({
        "message": "Art of Living Chatbot API is running",
        "environment": "production" if os.getenv('FLASK_ENV') == 'production' else "development"
    })

@routes.route("/ask", methods=["POST", "OPTIONS"])
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
        
        # Get the vectorstore (will be initialized only once)
        try:
            vectorstore = init_vectorstore()
            relevant_projects = vectorstore.similarity_search(user_input, k=3)
            print(f"Found {len(relevant_projects)} relevant projects")
        except Exception as e:
            print(f"Error in vector store: {str(e)}")
            print(traceback.format_exc())
            return jsonify({"error": "Error accessing database"}), 500

        projects_info = "\n\n".join([
            f"**{doc.metadata['title']}**\n- {doc.page_content}" for doc in relevant_projects
        ])

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
            model=MODEL_NAME,
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
        print(traceback.format_exc())
        return jsonify({"error": "Internal server error"}), 500
