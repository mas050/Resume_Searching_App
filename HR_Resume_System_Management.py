import re
import pandas as pd
import json
import streamlit as st
from PyPDF2 import PdfReader
import numpy as np
from scipy.spatial.distance import cosine
import google.generativeai as genai
import io

# Configure Google Gemini API
genai.configure(api_key="AIzaSyCOhsh-JWBd6B006GA0UgdIW6wRcNon7lk")

# Credentials for login
CREDENTIALS = {
    "GFT_USER": "GFT2024!"
}

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def login():
    """Login screen."""
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username in CREDENTIALS and CREDENTIALS[username] == password:
            st.session_state["logged_in"] = True
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")

# Function Definitions
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
        Segment the provided resume of a candidate into numbered chunks that adhere to the following guidelines:

        Identify natural breaks in the text, such as topic shifts, paragraph breaks, or sentence ends, similar to how a human would.
        Ensure each chunk maintains logical coherence with the content before and after it, using full sentences only.
        
        Avoid redundant or overlapping content between chunks, ensuring each contains unique information.
        Preserve any existing formatting, such as lists or subheadings, within each chunk to maintain the document's original structure.

        Do not modify, summarize or do anything to the candidate resume provided. You chunk it and output it as it is.

        Format your output as follows:

        Each chunk should be numbered sequentially, followed by a hyphen, and then the chunk itself.
        Provide only the ordered chunks — no introduction text, no conclusion text and no explanation of your task. 
        At the very beginning of the output, include "candidate_name: <Name>" where you will write the name of the candidate you find in the resume.
        If perfectly identical section of information is repeating itself across the document, only output it once. For example, footnote or header may repeat itself so only output once.

        Here's the format structure to follow:

        candidate_name: <Name>
        1 - Chunk 1
        2 - Chunk 2
        ...

        Here's the candidate resume to numbered chunks:
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
        
        new_data = pd.DataFrame({
            "pdf_file_name": [pdf_name] * len(chunks),
            "candidate_name": [candidate_name] * len(chunks),
            "resume_section_content": chunks,
            "embedded_chunk": embeddings,
        })
        historical_data = pd.concat([historical_data, new_data], ignore_index=True)
    
    return historical_data

# Main Streamlit App
if not st.session_state["logged_in"]:
    login()
else:
    st.title("AI-Powered Talent Discovery Assistant")

    # Sidebar Navigation
    app_mode = st.sidebar.radio(
        "Choose a function:",
        ["Resume Processing", "Resume Query"]
    )
    
    if app_mode == "Resume Processing":
        st.header("Resume Processing Application")
        uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        uploaded_csv = st.file_uploader("Upload historical CSV (optional)", type="csv")
        
        if uploaded_pdfs and st.button("Process Files"):
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
    
    elif app_mode == "Resume Query":
        st.header("Resume Query Application")
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            if 'embedded_chunk' in data.columns:
                data['embedded_chunk'] = data['embedded_chunk'].apply(deserialize_embedding)
            else:
                st.error("The uploaded CSV does not contain an 'embedded_chunk' column.")
                st.stop()
            
            user_question = st.text_input("Enter your question:")
            if user_question:
                user_embedding = generate_embedding(user_question)
                if user_embedding is not None:
                    data['similarity'] = data['embedded_chunk'].apply(
                        lambda chunk: cosine_similarity(user_embedding, chunk) if chunk is not None else -1
                    )
                    top_chunk_per_candidate = data.loc[data.groupby('candidate_name')['similarity'].idxmax()]
                    top_candidates = top_chunk_per_candidate.nlargest(10, 'similarity')
                    top_candidates['user_question'] = user_question
                    top_candidates = top_candidates[
                        ['candidate_name', 'pdf_file_name', 'user_question', 'resume_section_content', 'similarity']
                    ]
                    st.subheader("Top Matching Candidates")
                    st.dataframe(top_candidates)
                    csv_buffer = io.StringIO()
                    top_candidates.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_data,
                        file_name="top_candidates.csv",
                        mime="text/csv"
                    )
