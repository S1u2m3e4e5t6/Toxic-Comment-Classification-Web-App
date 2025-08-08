# 🧠 Toxic Comment Classification - NLP Project

A real-world NLP application that detects and classifies toxic comments from social media into categories like **toxic**, **insult**, **threat**, **obscene**, **identity hate**, and **severe toxic**. This is a multi-label text classification problem using deep learning models such as **BERT**.

🌐 **Live Demo:** [Streamlit App Link Here](#)

---

## 🚀 Project Overview

This project was inspired by Kaggle's **Jigsaw Toxic Comment Classification Challenge** and aims to help automate **content moderation** for online platforms. It uses a pre-trained **Transformer-based model** fine-tuned on toxic comment datasets.

---

## ✅ Key Features

- Multi-label classification using **BERT/DistilBERT**
- Handles **imbalanced datasets** with techniques like oversampling & class weighting
- Clean, user-friendly **Streamlit interface** for real-time predictions
- Integrated **visualization** of model confidence scores
- Preprocessing includes:
  - Removing special characters and stopwords
  - Tokenization and padding
  - Handling misspellings and bad tokens

---

## 📊 Dataset

- Source: [Jigsaw Toxic Comment Classification Dataset (Kaggle)](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- ~160,000+ comments with 6 possible toxicity labels

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Transformers (HuggingFace), scikit-learn, pandas, numpy, Streamlit
- **Model:** BERT / DistilBERT (fine-tuned)
- **Deployment:** Streamlit (Live UI), Docker (optional)
- **Evaluation Metrics:** Precision, Recall, F1 Score (macro & micro avg)

---

## 📁 Project Structure

├── app.py # Streamlit interface


├── model/


│ └── toxic_bert_model.pt # Trained model


├── data/


│ └── comments.csv # Training data


├── utils/


│ └── preprocessing.py # Cleaning & tokenizer functions


├── requirements.txt # Dependencies


└── README.md


![Status](https://img.shields.io/badge/status-in--progress-yellow)
### 📅 Project Roadmap

- [x] Idea & Planning
- [x] Tech Stack Finalized
- [ ] Core Features Development (Ongoing)
- [ ] Testing & Debugging
- [ ] Documentation
- [ ] Final Release (Planned before September 2025)


---

## 🧪 How to Run Locally

```bash
git clone https://github.com/yourusername/toxic-comment-classifier.git
cd toxic-comment-classifier
pip install -r requirements.txt
streamlit run app.py




---

Let me know if you’d like me to generate a `requirements.txt` file or `app.py` starter code as well.






