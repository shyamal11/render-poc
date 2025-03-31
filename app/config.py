import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
MODEL_NAME = os.getenv("MODEL_NAME")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
EXCEL_PATH = "Data Sample for Altro AI.xlsx"
SHEET_NAME = "REAL and Mocked up Data for POC"
