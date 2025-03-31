import pandas as pd
from langchain_core.documents import Document
from app.config import EXCEL_PATH, SHEET_NAME

def load_documents():
    xls = pd.ExcelFile(EXCEL_PATH)
    data = pd.read_excel(xls, SHEET_NAME).fillna("")
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

    return documents
