# AI-Powered Resume Ranker

# Install required packages:
# pip install sentence-transformers pdfplumber streamlit

import os
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
from io import BytesIO

# Load pre-trained BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Styling ---
st.set_page_config(page_title="AI Resume Ranker", page_icon="📄", layout="centered")

# Custom CSS
st.markdown("""
    <style>
    .title-style {
        font-size: 40px;
        font-weight: bold;
        color: #4A90E2;
    }
    .subtitle-style {
        font-size: 20px;
        color: #333;
        margin-bottom: 20px;
    }
    .score-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        font-size: 18px;
    }
    </style>
""", unsafe_allow_html=True)

# Title Section
st.markdown('<div class="title-style">📄 AI-Powered Resume Ranker</div>', unsafe_allow_html=True)

st.markdown('<div class="subtitle-style">Rank resumes based on a job description using AI and BERT embeddings.</div>', unsafe_allow_html=True)

# --- Functions ---
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def rank_resumes(job_description, resume_texts):
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    results = []
    for name, text in resume_texts.items():
        embedding = model.encode(text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(jd_embedding, embedding).item()
        results.append((name, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# --- Input UI ---
st.subheader("Step 1: Enter Job Description")
job_description = st.text_area("Paste the job description here:", height=200)

st.subheader("Step 2: Upload Resumes")
uploaded_files = st.file_uploader("Upload PDF resumes:", accept_multiple_files=True, type=["pdf"])

if st.button("🚀 Rank Resumes"):
    if job_description and uploaded_files:
        resume_texts = {}
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resume_texts[file.name] = text

        with st.spinner("Analyzing and ranking resumes..."):
            rankings = rank_resumes(job_description, resume_texts)

        st.success("✅ Ranking complete!")
        st.subheader("📊 Ranked Resumes:")
        for i, (name, score) in enumerate(rankings, 1):
            st.markdown(f"<div class='score-box'>{i}. <strong>{name}</strong> — Similarity Score: <strong>{score:.4f}</strong></div>", unsafe_allow_html=True)

    else:
        st.warning("⚠️ Please provide a job description and upload at least one resume.")

# --- Footer ---
st.markdown("""
---
**Project by [Your Name]** | Showcase this project on [GitHub](https://github.com/yourusername/resume-ranker)
""")
