import re
import pandas as pd
import json
import streamlit as st
from PyPDF2 import PdfReader
import google.generativeai as genai

# Configure Google Gemini
genai.configure(api_key="AIzaSyCOhsh-JWBd6B006GA0UgdIW6wRcNon7lk")

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    try:
        reader = PdfReader(pdf_file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return ""

def get_text_embeddings(text_from_pdf):
    """Generate embeddings for text using Google's 'text-embedding-004' model."""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text_from_pdf
        )
        embeddings = result['embedding']
        return embeddings
    except Exception as e:
        st.error(f"Error generating embeddings: {e}")
        return None

def llm_section_identification(text_from_pdf):
    """Identify sections in text using LLM."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    llm_prompt = f"""
        Segment the provided context into numbered chunks that adhere to the following guidelines:

        Identify natural breaks in the text, such as topic shifts, paragraph breaks, or sentence ends, similar to how a human would.
        Ensure each chunk maintains logical coherence with the content before and after it, using full sentences only.

        Format your output as follows:

        candidate_name: <Name>
        1 - Chunk 1
        2 - Chunk 2
        ...

        Here's the context to numbered chunks:
        {text_from_pdf}
    """
    try:
        response = model.generate_content(llm_prompt)
        return response.text
    except Exception as e:
        st.error(f"Error processing text with LLM: {e}")
        return ""

def process_uploaded_files(pdf_files, csv_file):
    """Process uploaded PDF and CSV files."""
    # Check if a valid CSV was uploaded
    if csv_file:
        try:
            historical_data = pd.read_csv(csv_file)
            required_columns = {"pdf_file_name", "candidate_name", "resume_section_content", "embedded_chunk"}
            if not required_columns.issubset(historical_data.columns):
                st.error(f"The uploaded CSV must contain the columns: {required_columns}")
                return None
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return None
    else:
        historical_data = pd.DataFrame(columns=["pdf_file_name", "candidate_name", "resume_section_content", "embedded_chunk"])
    
    # Process each PDF
    for pdf_file in pdf_files:
        pdf_name = pdf_file.name
        st.info(f"Processing: {pdf_name}")
        pdf_text = extract_text_from_pdf(pdf_file)
        
        if not pdf_text:
            st.warning(f"Skipping {pdf_name} due to extraction error.")
            continue
        
        section_text = llm_section_identification(pdf_text)
        if not section_text:
            st.warning(f"Skipping {pdf_name} due to LLM processing error.")
            continue
        
        candidate_name_match = re.search(r"candidate_name:\s*(.+)", section_text)
        if not candidate_name_match:
            st.warning(f"Candidate name not found in {pdf_name}.")
            continue
        candidate_name = candidate_name_match.group(1).strip()
        
        chunks = re.findall(r"\d+\s*-\s*(.+?)(?=\n\d+\s*-|$)", section_text, re.DOTALL)
        embeddings = [json.dumps(get_text_embeddings(chunk)) for chunk in chunks]
        
        # Append to historical data
        new_data = pd.DataFrame({
            "pdf_file_name": [pdf_name] * len(chunks),
            "candidate_name": [candidate_name] * len(chunks),
            "resume_section_content": chunks,
            "embedded_chunk": embeddings,
        })
        historical_data = pd.concat([historical_data, new_data], ignore_index=True)
    
    return historical_data

# Streamlit App
st.title("Resume Processing Application")

st.sidebar.header("Upload Files")
uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)
uploaded_csv = st.sidebar.file_uploader(
    "Upload historical CSV (optional)", type="csv"
)

if uploaded_pdfs:
    if st.button("Process Files"):
        result_df = process_uploaded_files(uploaded_pdfs, uploaded_csv)
        if result_df is not None:
            st.success("Processing completed!")
            st.dataframe(result_df)
            st.download_button(
                "Download Results as CSV",
                result_df.to_csv(index=False),
                file_name="processed_resumes.csv",
                mime="text/csv"
            )
else:
    st.info("Please upload PDF files to begin processing.")
