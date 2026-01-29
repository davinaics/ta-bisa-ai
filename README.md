# 📊 Customer Review Dashboard

An interactive **Streamlit-based dashboard** for analyzing Indonesian e-commerce customer reviews using:
- **Exploratory Data Analysis (EDA)**
- **Machine Learning (Random Forest Classifier)**
- **Retrieval-Augmented Generation (RAG) Chatbot**

This project analyzes product reviews from Tokopedia to understand customer sentiment and emotions, evaluate sales patterns, and provide an intelligent chatbot for review-based insights.

---

## 🗂️ Dataset

The dataset used in this project is **PRDECT-ID**, a comprehensive Indonesian product review dataset with sentiment and emotion annotations.  
The dataset was obtained from **Kaggle** and originally sourced from **Tokopedia**, covering **29 product categories**.

### Dataset Columns:
- `Category`  
- `Product Name`  
- `Location`  
- `Price`  
- `Overall Rating`  
- `Number Sold`  
- `Total Review`  
- `Customer Rating`  
- `Customer Review`  
- `Sentiment`  
- `Emotion`

Source: https://www.kaggle.com/datasets/jocelyndumlao/prdect-id-indonesian-emotion-classification 

---

## 🔄 Project Stages

### 1️⃣ Data Preprocessing

Two types of data were processed:

#### 📌 Structured Data
- Missing value checking  
- Duplicate removal  
- Outlier detection  
- **Output:** `cleaned_dataset_full.csv` (used for EDA)

#### 📌 Unstructured Text Data (Customer Reviews)
- Lowercasing  
- URL, mention, and number removal  
- Tokenization  
- Stopword removal  
- **Output:** `cleaned_dataset_model.csv` (used for modeling & chatbot)

---

### 2️⃣ Analysis and Modeling

- Text representation using **TF-IDF Vectorizer**
- Sentiment classification using **Random Forest Classifier**
- Model evaluation using:
  - Accuracy  
  - Precision  
  - Recall  
  - F1-score  
  - Confusion Matrix  
- Integration of **Large Language Model (LLM)** from **DeepSeek (via OpenRouter)** for chatbot RAG functionality

**Model Artifacts:**
- `tfidf_vectorizer.pkl`  
- `rf_model.pkl`

These models are loaded directly into the Streamlit application.

---

## 🖥️ Dashboard Implementation (Streamlit)

The dashboard consists of 3 main tabs:

### 📊 EDA Tab
- Sentiment distribution  
- Sales distribution  
- KPI cards (total products, categories, sales, reviews, average rating)  
- Top 5 cities by sales  
- Top 5 categories by sales  
- Top 5 best-selling products  
- Top customer emotions  

---

### 🤖 Modelling Tab
- Model evaluation results  
- Classification report  
- Confusion matrix  
- WordCloud of most frequent words  
- New review sentiment prediction  

---

### 💬 Chatbot (RAG Tab)
- Answers user questions based on real customer reviews  
- Uses:
  - TF-IDF + cosine similarity for retrieval  
  - LLM (DeepSeek via OpenRouter) for answer generation  
- Displays:
  - Summarized answer  
  - Example original reviews used as references  

---

## 🗂️ Project Structure
```bash
TugasAkhir/
│
├── dashboard.py
├── cleaned_dataset_full.csv
├── cleaned_dataset_model.csv
├── rf_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── .streamlit/
│ └── secrets.toml
└── README.md
```

---

## ⚙️ Installation & Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run dashboard.py
```

## 🔑 API Configuration (Chatbot)
1. Create the file:
```bash
.streamlit/secrets.toml
```

2. Add your API key:
```bash
OPENROUTER_API_KEY="YOUR_API_KEY"
```




