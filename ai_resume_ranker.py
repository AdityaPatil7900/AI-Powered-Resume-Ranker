# AI-Powered Resume Ranker -

# Required packages:
# pip install streamlit sentence-transformers pdfplumber

import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer, util

# Load Model (cached for speed)
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ---- Page Config ----
st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="🧾",
    layout="centered"
)

# ---- Custom Styling ----
st.markdown("""
<style>
body {
    background-color: #f8fafc;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Title */
.title-style {
    font-size: 45px;
    font-weight: 800;
    color: #4A90E2;  /* Dark Navy for professional impact */
    text-align: center;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle-style {
    font-size: 19px;
    text-align: center;
    color: #475569;
    margin-bottom: 30px;
}

/* Score Card */
.score-box {
    background: linear-gradient(135deg, #1d4ed8, #2563eb);
    color: white;
    padding: 14px;
    border-radius: 8px;
    margin: 10px 0;
    font-size: 17px;
    font-weight: 500;
    transition: 0.3s ease;
}
.score-box:hover {
    background: linear-gradient(135deg, #1e40af, #1d4ed8);
}

/* Upload Box */
.upload-box {
    border: 2px dashed #94a3b8;
    padding: 20px;
    text-align: center;
    border-radius: 8px;
    background-color: #f1f5f9;
    margin-bottom: 20px;
}

/* Button */
button[data-baseweb="button"] {
    background-color: #2563eb;
    color: white;
    font-weight: 600;
    border-radius: 8px;
}
button[data-baseweb="button"]:hover {
    background-color: #1d4ed8;
}

/* Hide Streamlit footer */
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown('<div class="title-style">🧾 AI-Powered Resume Ranker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle-style">Professionally rank resumes based on job description relevance using AI embeddings</div>', unsafe_allow_html=True)

# ---- PDF Extraction ----
def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ---- Ranking Logic ----
def rank_resumes(job_description, resume_texts):
    jd_embedding = model.encode(job_description, convert_to_tensor=True)
    results = []
    for name, text in resume_texts.items():
        embedding = model.encode(text, convert_to_tensor=True)
        score = util.pytorch_cos_sim(jd_embedding, embedding).item()
        results.append((name, score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ---- Inputs ----
st.subheader("Step 1: Provide Job Description 🖋")
job_description = st.text_area(
    "Enter detailed job description:",
    height=180,
    placeholder="Describe responsibilities, skills, and requirements..."
)

st.subheader("Step 2: Upload Candidate Resumes 📂")
uploaded_files = st.file_uploader(
    "Upload multiple PDF resumes",
    accept_multiple_files=True,
    type=["pdf"]
)

# ---- Rank Button ----
if st.button("📊 Analyze & Rank Resumes", use_container_width=True):
    if job_description and uploaded_files:
        resume_texts = {}
        for file in uploaded_files:
            text = extract_text_from_pdf(file)
            resume_texts[file.name] = text

        with st.spinner("Evaluating resumes... Please wait."):
            rankings = rank_resumes(job_description, resume_texts)

        st.success("✔ Ranking Completed Successfully!")
        st.subheader("Ranked Resumes:")

        for i, (name, score) in enumerate(rankings, 1):
            st.markdown(
                f"<div class='score-box'>{i}. {name} — Relevance Score: {score:.2%}</div>",
                unsafe_allow_html=True
            )

    else:
        st.warning("⚠ Please enter a job description and upload at least one resume.")

# ---- Footer ----
st.markdown("""
---
<div style='text-align:center; font-size:15px; color:#475569'>
Developed with precision using <b>Streamlit</b> & <b>BERT AI</b>  
Showcase on <a href='https://github.com/yourusername/resume-ranker' target='_blank'>GitHub</a>
</div>
""", unsafe_allow_html=True)
