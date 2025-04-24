# 📰 Automated News Categorization using Deep Learning and Machine Learning

This project tackles the challenge of categorizing news articles into predefined categories using a combination of classical machine learning and modern deep learning techniques. Built as part of CSC 8980: Topics in Computer Science, this solution emphasizes scalability, real-time adaptability, and robustness against unstructured text data.

## 📌 Problem Statement

News content varies in style and vocabulary, making it hard to classify with rule-based methods. Traditional models often fail to understand ambiguous or overlapping category boundaries. This project proposes a scalable, ML-powered solution to automate this task using both statistical and semantic techniques.

---

## 📊 Dataset

- **AG News Dataset** (10,000 samples used)
- Features: `Title`, `Description`
- Target Categories:
  - World News
  - Sports News
  - Business News
  - Science & Technology

---

## 🧪 Methodology

### 1. **Preprocessing**
- HTML tag and special character removal
- Lowercasing
- Custom stopword removal (preserved key negations like *not*, *no*)
- Tokenization using NLTK
- Lemmatization
- Word cloud visualization for exploratory analysis

### 2. **Feature Extraction Techniques**
- **Bag of Words (BoW)**
- **TF-IDF**
- **Word2Vec**
- **BERT Embeddings**

### 3. **Models Used**
- Classical ML: Naive Bayes, Logistic Regression, Random Forest, SVM, KNN, Decision Tree
- Deep Learning: Fine-tuned `bert-base-uncased` using Hugging Face Transformers

---

## 🧠 Model Training & Evaluation

| Technique     | Model                | Accuracy (%) |
|---------------|----------------------|--------------|
| BoW           | Multinomial NB       | 88.07        |
| TF-IDF        | Multinomial NB       | 88.20        |
| Word2Vec      | Logistic Regression  | 70.77        |
| BERT embedding| Logistic Regression  | 87.53        |
| **BERT**      | Fine-tuned Transformer| **94.2**     |

- Evaluation Metrics: **Accuracy**, **Precision**, **Recall**, **F1-score**

---

## 🚀 Key Highlights

- **Fine-tuned BERT** model achieved the highest performance with 94.2% accuracy.
- Used Hugging Face Transformers for embedding generation and classification.
- Classical models provided solid baselines for comparison and insight.
- Demonstrated the value of combining traditional and transformer-based NLP methods.

---

## ⚙️ How to Run

1. Execute cells sequentially (Preprocessing → Feature Engineering → Model Training).
2. Modify sections to test different models or vectorizers.

---

## 📚 Tech Stack

- **Languages**: Python
- **Libraries**: Scikit-learn, NLTK, Hugging Face Transformers, Pandas, Matplotlib, NumPy, TensorFlow (for BERT fine-tuning)

---

## 📜 License

This project is created for academic and educational use as part of the graduate curriculum.

---

