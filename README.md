# 📄 AI-Powered Resume Ranker

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

> 🚀 A smart, NLP-based AI application that ranks resumes based on how well they match a job description using Sentence-BERT embeddings. Perfect for HR teams, recruiters, and your portfolio!

---

## ✨ Features
- 📥 Upload multiple resumes (PDF format)
- 📝 Enter any job description
- 🤖 Leverages BERT embeddings to semantically compare resumes
- 📊 Ranks resumes by relevance
- 💬 Clean, interactive UI with Streamlit
- 🎯 Ideal for portfolio projects, HR tech tools, and interview screening automation

---

## 🧠 How It Works
1. **Job description** is embedded using a BERT model.
2. **Each resume** is parsed and embedded the same way.
3. **Cosine similarity** is computed between job description and each resume.
4. Results are **ranked from highest to lowest** match.

---

## 🛠️ Tech Stack
| Technology | Role |
|-----------|------|
| Python | Programming Language |
| Streamlit | Web App Interface |
| Sentence-BERT | Embedding & Semantic Search |
| pdfplumber | PDF Text Extraction |

---

## 📸 Screenshots
| Home Screen | Ranked Output |
|-------------|---------------|
| ![home](screenshots/home.png) | ![output](screenshots/output.png) |

---

## 🚀 Getting Started

### 🔧 Installation
```bash
# Clone the repo
git clone (https://www.linkedin.com/in/aditya-patil-aj7900/)ai-resume-ranker.git
cd ai-resume-ranker

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## 📁 Project Structure
```
ai-resume-ranker/
├── app.py                # Main Streamlit application
├── requirements.txt      # Dependencies
├── README.md             # Project documentation
└── screenshots/          # UI screenshots
    ├── home.png
    └── output.png
```

---

## ✅ Use Cases
- Recruiters automating initial resume filtering
- Job boards integrating smarter search
- Students showcasing real-world AI applications
- Developers exploring NLP/semantic similarity projects

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).

---
## 👤 Author
**[Aditya Patil]**  
🔗 [LinkedIn](https://www.linkedin.com/in/aditya-patil-aj7900/)<br> 
🔗 [GitHub](https://github.com/AdityaPatil7900)

---
  
> ✨ Star this repo if you find it useful. Contributions are welcome!
