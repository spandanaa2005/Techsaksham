import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf = PdfReader(file)
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or "" # Added or "" to handle pages with no text
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return ""

# Function to rank resumes based on job description
def rank_resumes(job_description, resumes):
    documents = [job_description] + resumes  # Create a list of all documents
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(documents)
    job_description_vector = vectors[0]
    resume_vectors = vectors[1:]
    cosine_similarities = cosine_similarity(job_description_vector, resume_vectors).flatten() #added flatten()
    return cosine_similarities

# Streamlit app
st.title("AI Resume Screening & Candidate Ranking System")
st.header("Job Description")
job_description_text = st.text_area("Enter the job description")

st.header("Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if job_description_text and uploaded_files: #changed uploaded_files to job_description
    st.header("Ranking Resumes")
    resumes = []
    resume_names = [] #store resume names
    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        if text:  # Only add if text extraction was successful
            resumes.append(text)
            resume_names.append(file.name) #add resume names
    if resumes:
        scores = rank_resumes(job_description_text, resumes)
        results = pd.DataFrame({"Resume": resume_names, "Score": scores})
        results = results.sort_values(by="Score", ascending=False)
        st.write(results)
    else:
        st.warning("No resumes were successfully processed.")
else:
    st.info("Please enter a job description and upload resumes.")