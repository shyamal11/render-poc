import pandas as pd
from langchain.docstore.document import Document
from app.config import EXCEL_PATH, SHEET_NAME
import os

def load_data():
    """
    Load data from Excel file and convert to LangChain documents
    """
    try:
        # Get the data directory path
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        excel_path = os.path.join(data_dir, EXCEL_PATH)
        
        # Read Excel file
        df = pd.read_excel(excel_path, sheet_name=SHEET_NAME)
        
        # Convert DataFrame to documents
        documents = []
        for _, row in df.iterrows():
            # Create document content
            content = f"""
            Title: {row.get('title', '')}
            Description: {row.get('description', '')}
            Location: {row.get('location', '')}
            Budget: {row.get('budget', '')}
            Skills Required: {row.get('skills_required', '')}
            """
            
            # Create metadata
            metadata = {
                'title': row.get('title', ''),
                'location': row.get('location', ''),
                'budget': row.get('budget', ''),
                'skills_required': row.get('skills_required', '')
            }
            
            # Create document
            doc = Document(
                page_content=content,
                metadata=metadata
            )
            documents.append(doc)
        
        print(f"Loaded {len(documents)} documents from Excel file")
        return documents
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise
