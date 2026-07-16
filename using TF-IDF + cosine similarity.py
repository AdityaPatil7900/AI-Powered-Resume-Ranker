# ===========================================
# AI Resume Ranker using TF-IDF
# Author: Your Name
# ===========================================

import pdfplumber
import streamlit as st
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download("stopwords")

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI Resume Ranker",
    page_icon="🧾",
    layout="centered"
)

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

body{
    background-color:#f8fafc;
    font-family:'Segoe UI';
}

.title-style{
    font-size:45px;
    font-weight:800;
    color:#2563eb;
    text-align:center;
}

.subtitle-style{
    font-size:18px;
    color:#475569;
    text-align:center;
    margin-bottom:25px;
}

.score-box{
    background:linear-gradient(135deg,#2563eb,#1d4ed8);
    color:white;
    padding:14px;
    border-radius:8px;
    margin-top:10px;
    font-size:17px;
    font-weight:600;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Heading
# -----------------------------
st.markdown(
    '<div class="title-style">🧾 AI Resume Ranker</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle-style">Rank resumes using TF-IDF & Cosine Similarity</div>',
    unsafe_allow_html=True
)

# -----------------------------
# PDF Text Extraction
# -----------------------------
def extract_text_from_pdf(pdf_file):

    text = ""

    with pdfplumber.open(pdf_file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# -----------------------------
# Text Preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = text.lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)


# -----------------------------
# Resume Ranking
# -----------------------------
def rank_resumes(job_description, resume_texts):

    documents = [preprocess(job_description)]

    filenames = []

    for filename, resume in resume_texts.items():

        documents.append(preprocess(resume))

        filenames.append(filename)

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(documents)

    jd_vector = tfidf_matrix[0]

    resume_vectors = tfidf_matrix[1:]

    similarity = cosine_similarity(
        jd_vector,
        resume_vectors
    )[0]

    rankings = []

    for i in range(len(similarity)):
        rankings.append(
            (
                filenames[i],
                similarity[i]
            )
        )

    rankings.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return rankings


# -----------------------------
# User Input
# -----------------------------
st.subheader("Step 1 : Enter Job Description")

job_description = st.text_area(
    "Paste the Job Description",
    height=180,
    placeholder="Enter responsibilities, required skills and qualifications..."
)

st.subheader("Step 2 : Upload Candidate Resumes")

uploaded_files = st.file_uploader(
    "Upload multiple PDF resumes",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# Analyze Button
# -----------------------------
if st.button(
    "📊 Analyze & Rank Resumes",
    use_container_width=True
):

    if job_description and uploaded_files:

        resume_texts = {}

        for file in uploaded_files:

            resume_texts[file.name] = extract_text_from_pdf(file)

        with st.spinner("Analyzing resumes..."):

            rankings = rank_resumes(
                job_description,
                resume_texts
            )

        st.success("Ranking Completed Successfully!")

        st.subheader("Resume Ranking")

        for rank, (name, score) in enumerate(rankings, start=1):

            st.markdown(
                f"""
                <div class='score-box'>
                {rank}. {name}<br>
                Match Score : {score*100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

    else:

        st.warning(
            "Please enter a Job Description and upload at least one resume."
        )

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
---
<div style='text-align:center;color:gray;font-size:14px;'>

Developed using
<b>Python</b>,
<b>Streamlit</b>,
<b>TF-IDF</b>,
<b>Cosine Similarity</b>

</div>
""", unsafe_allow_html=True)
