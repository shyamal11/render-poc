from flask import Flask
from flask_cors import CORS
from app.routes import routes
from app.vector_store import init_vectorstore
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get environment
is_production = os.getenv('FLASK_ENV') == 'production'
logger.info(f"Running in {'production' if is_production else 'development'} mode")

# Configure CORS based on environment
if is_production:
    # Production CORS settings
    CORS(app, resources={
        r"/*": {
            "origins": [
                "https://*.netlify.app",  # All Netlify domains
                "https://your-site-name.netlify.app"  # Your specific Netlify domain
            ],
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type"]
        }
    })
else:
    # Development CORS settings - more permissive
    CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize vector store during app startup
logger.info("Starting application initialization...")
try:
    init_vectorstore()
    logger.info("Vector store initialization complete!")
except Exception as e:
    logger.error(f"Error initializing vector store: {str(e)}")

app.register_blueprint(routes)

if __name__ == '__main__':
    # Get port dynamically from Render's environment or default to 8000
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Application will run on port: {port}")
    app.run(host='0.0.0.0', port=port)
