import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import google.generativeai as genai
import json

# Configure the Google Gemini API key
genai.configure(api_key="AIzaSyCOhsh-JWBd6B006GA0UgdIW6wRcNon7lk")

# Function to compute cosine similarity
def cosine_similarity(vec1, vec2):
    return 1 - cosine(vec1, vec2)

# Function to generate embedding for a text using Gemini
def generate_embedding(text):
    """
    Generate embedding using Google's text-embedding-004 model.
    
    Args:
        text (str): Input text to generate the embedding for.
    
    Returns:
        np.array: The embedding as a NumPy array, or None if an error occurs.
    """
    try:
        # Call the Google embedding model
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text
        )
        
        # Extract the embedding values
        embedding = result['embedding']

        return np.array(embedding)  # Convert embedding to a NumPy array
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

# Function to deserialize embeddings from JSON strings
def deserialize_embedding(json_string):
    try:
        # Parse the JSON string into a Python list
        embedding_list = json.loads(json_string)
        # Convert to a NumPy array
        return np.array(embedding_list)
    except (json.JSONDecodeError, TypeError) as e:
        st.error(f"Error deserializing embedding: {e}")
        return None

# Streamlit UI
st.title("Resume Query App")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV into a DataFrame
    data = pd.read_csv(uploaded_file)

    # Deserialize 'embedded_chunk' column
    if 'embedded_chunk' in data.columns:
        data['embedded_chunk'] = data['embedded_chunk'].apply(deserialize_embedding)
    else:
        st.error("The uploaded CSV does not contain an 'embedded_chunk' column.")
        st.stop()

    # User-provided question
    user_question = st.text_input("Enter your question:")

    if user_question:
        # Generate embedding for the user's question
        user_embedding = generate_embedding(user_question)
        if user_embedding is None:
            st.error("Failed to generate embedding for the user's question.")
        else:
            # Compute cosine similarity for each embedded_chunk
            data['similarity'] = data['embedded_chunk'].apply(
                lambda chunk: cosine_similarity(user_embedding, chunk) if chunk is not None else -1
            )

            # Find the top 1 chunk for each candidate_name
            top_chunk_per_candidate = data.loc[data.groupby('candidate_name')['similarity'].idxmax()]

            # Sort candidates by the similarity of their top chunk in descending order
            top_candidates = top_chunk_per_candidate.nlargest(10, 'similarity')

            # Add the user question column
            top_candidates['user_question'] = user_question

            # Reorder columns: candidate_name, user_question, resume_section_content, similarity
            top_candidates = top_candidates[
                ['candidate_name', 'pdf_file_name', 'user_question', 'resume_section_content', 'similarity']
            ]

            # Display the results
            st.subheader("Top Matching Candidates")
            st.dataframe(top_candidates)

else:
    st.info("Please upload a CSV file to get started.")
