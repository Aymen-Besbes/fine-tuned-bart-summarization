#  🧠 Fine-Tuned BART for Abstractive Text Summarization (AI & LLM Domain)

This project demonstrates an end-to-end **abstractive summarization pipeline** using a fine-tuned `facebook/bart-large-cnn` model on a domain-specific dataset about **AI, LLMs, and NLP technologies**. 

It includes:
1. Custom dataset preparation  
2. Exploratory Data Analysis (EDA)  
3. BART fine-tuning using Hugging Face Transformers
4.  ROUGE-based evaluation
5.  Real-time Gradio demo app 
---

## 📁 Project Structure
```plaintext
fine-tuned-bart-summarization/
│
├── notebooks/
| ├── EDA.ipynb 
│ └── fine-tune.ipynb
│
├── scripts/
│ ├── fine_tune.py # Fine-tuning & evaluation script
│ └── inference_demo.py # Gradio app script
│
├── data/
│ ├── ai_llm_dataset.csv # Collected dataset
│ └── Enhanced_ai_llm_dataset.csv # Final training dataset
│
├── fine_tuned_bart_model/ # Exported fine-tuned model
├── requirements.txt
├── .gitignore
└── README.md 
```
---

## 📊 Dataset Overview

The project uses two datasets stored in the `data/` folder:

### 🔹 `ai_llm_dataset.csv`  
- Raw manually curated dataset  
- 20 text-summary pairs related to AI and LLMs  
- Columns: `text`, `summary`

### 🔸 `Enhanced_ai_llm_dataset.csv`  
- Feature-enriched version of the above  
- Columns:
  - `text`, `summary`
  - `text_length`, `summary_length`
  - `text_word_count`, `summary_word_count`
  - `text_summary_similarity`

These features were used to determine max sequence lengths, visualize dataset structure, and improve training feedback.

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook [`notebooks/eda_ai_llm_dataset.ipynb`](notebooks/eda_ai_llm_dataset.ipynb) contains:

- Distribution plots for text/summary lengths and word counts  
- Word frequency analysis and word clouds  
- Text-summary similarity histograms  
- Correlation matrix for engineered features  

EDA helped guide:
- Truncation & padding strategy
- Model input configuration
- Evaluation design
- ### 📊 Key Visuals

#### 🔹 Word Cloud (Text Columns)

![Word Cloud](images/wordcloud_texts.png)

#### 🔹 Distribution of Text Lengths

![Text Lengths](images/text_length_distribution.png)

#### 🔹 Text-Summary Similarity Scores

![Similarity Histogram](images/summary_similarity_distribution.png)

---

## ⚙️ Fine-Tuning Details

We fine-tuned `facebook/bart-large-cnn` using Hugging Face’s `Seq2SeqTrainer` on the enhanced dataset.

### ✅ Training Arguments Used

| Argument                | Value             |
|------------------------|--------------------|
| `num_train_epochs`     | 10                 |
| `per_device_train_batch_size` | 2         |
| `per_device_eval_batch_size` | 2           |
| `eval_steps`           | 10                 |
| `logging_steps`        | 5                  |
| `weight_decay`         | 0.01               |
| `predict_with_generate`| ✅ True            |
| `report_to`            | "none" (W&B off)   |

Tokenization `max_length` values were dynamically set using the 75th percentile of tokenized input/target lengths.

---

## 📈 ROUGE Scores (Fine-Tuned Model)

| Metric     | Score   |
|------------|---------|
| ROUGE-1    | 0.3395  |
| ROUGE-2    | 0.0968  |
| ROUGE-L    | 0.2486  |
| ROUGE-Lsum | 0.2468  |


---

## 🚀 Run the Gradio Summarization App

You can test the model in real time using a simple UI.

### 💡 Steps:

1. **Install requirements**  
   Run from project root:

   ```bash
   pip install -r requirements.txt
2. ** Launch the app**
   ```bash
   cd scripts
   python inference_demo.py
---
## 📬 Contact
Author: Aymen Besbes & Aws Gandouz Email: Aymen.besbes@outlook.com | Aymen.besbes@ensi-uma.tn

LinkedIn: https://www.linkedin.com/in/aymen-besbes-158837245/

---

## 📅 Project Timeline
Original creation date: June - July 2025
Upload to GitHub: July 2025
