# 📱 Multilingual Mobile App Reviews — Sentiment Analysis (2025)

A multilingual sentiment analysis pipeline built on the **"Multilingual Mobile App Reviews Dataset 2025"** (Kaggle), using **XLM-Roberta** for transformer-based fine-tuning.  
Supports over 100 languages — no translation needed.

---

## 📋 Project Overview
User reviews in mobile app stores come from a global audience, in many languages, and with various app categories.  
Our goal is to **predict sentiment** (Negative, Neutral, Positive) from review text — regardless of language — while handling class imbalance, noisy metadata, and multilingual preprocessing challenges.

---

## 🚀 Features
- **Automatic Kaggle dataset import** — ready to run in Kaggle Notebook.
- **Multilingual tokenization** with `xlm-roberta-base`.
- **Custom sentiment labeling** from numeric ratings:
  - `rating <= 2` → Negative (0)
  - `rating = 3` → Neutral (1)
  - `rating >= 4` → Positive (2)
- **EDA** across languages and sentiments.
- **Data cleaning pipeline** (lowercasing, HTML/URL removal, Unicode-safe filtering).
- **Transfer learning via fine-tuning** (PyTorch + Hugging Face `transformers`).
- **Evaluation metrics**: Accuracy, Precision, Recall, **Macro-F1**.
- Planned: Confusion matrix & per-language performance breakdown.

---

## 📂 Dataset
*[*Source:** [Multilingual Mobile App Reviews Dataset 2025 — Kaggle](https://kaggle.com/)  
**Shape:** 2,514 rows × 15 columns  
**Languages:** ~20+, top = Russian, Polish, Korean (English not dominant).  
**Missing data:**  
- `review_text`: 2.3%
- `rating`: 1.47%
- `user_gender`: 23%

---

## 📊 Exploratory Data Analysis (Step 1–3)
- Found **class imbalance** (Negative > Positive > Neutral).
- Metadata (`app_category`) is noisy (e.g., Netflix tagged as Dating).
- Languages are unevenly distributed.  
  → Decided to **keep text in original language** and use multilingual embeddings instead of translation.

---

## 🧹 Preprocessing (Step 4)
- Removed HTML tags and URLs.
- Lowercased text (without breaking non-Latin scripts).
- Tokenized to `input_ids` and `attention_mask` with **max_length = 128** using `AutoTokenizer.from_pretrained("xlm-roberta-base")`.

---

## 🏋️ Model Training (Step 5)
- **Architecture:**
  - Base: `xlm-roberta-base`
  - Head: Sequence classification (3 classes)
- **Loss:** CrossEntropy
- **Training arguments:**  
  - epochs = 3 (planned ↑ for better performance)
  - batch_size = 16
  - learning_rate = 2e-5 (planned ↓ for stability)
  - weight_decay = 0.01
- **Evaluation:** every epoch; `metric_for_best_model="eval_f1_macro"`
- **Results so far:**  
  - Accuracy ≈ 0.48  
  - Macro-F1 ≈ 0.217  
  → Model not learning effectively; next steps focus on rebalancing & tuning.

---

## 📈 Next Improvements
- Increase epochs (8–10) and adjust learning rate (1e-5).
- Use stratified splits for balanced train/val sets.
- Apply class weights to combat imbalance.
- Increase `max_length` to 256 for long reviews.
- Per-language performance breakdown (Step 6).
- Confusion matrix visualization.

---

## 🛠 Tech Stack
- **Language:** Python 3.x
- **Libraries:**  
  - `transformers` 4.52.x  
  - `torch`  
  - `pandas`, `numpy`  
  - `scikit-learn`  
  - `matplotlib`, `seaborn`
- **Platform:** Kaggle Notebook (GPU: T4)

---

## 📜 License
MIT License — feel free to fork, adapt, and improve.

---

## ✍️ Author
Developed by Ali Abdollahi — 2025.

---
