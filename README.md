# 🧴 Skincare Compatibility Checker

A machine learning app that helps users determine whether a Sephora skincare 
product is compatible with their current routine, before they buy it.  

## Team Members
- Aishah Abdul-Hakeemn (aaa279)
- Siri Reddy (smr355)
- Jaydon Akintlotan (jaa374)

---

## Overview

Many skincare products contain ingredient combinations that interact poorly with 
existing routines, leading to irritation or reduced effectiveness. This project 
uses the Sephora Products and Skincare Reviews dataset to:

- Predict product ratings using ingredient and sentiment features
- Score ingredient compatibility between a user's routine and a new product
- Summarize real customer reviews for any selected product

---

## App Features

- 🔍 **Product Search** — search across 2,400+ Sephora skincare products
- 🧪 **Compatibility Score** — Low / Moderate / High Risk based on ingredient co-occurrence patterns
- ⭐ **Rating** — average rating and review count from verified buyers
- 📝 **Review Summary** — extractive summary from the most helpful reviews
- 📊 **Side-by-side Comparison** — compare up to 3 products at once

---

## Project Structure  
 ```
skincare-ai/  
├── notebooks/    
│   ├── 01_data_cleaning_and_ingredient_analysis.ipynb  
│   ├── 02_sentiment_analysis.ipynb  
│   ├── 03_model_training.ipynb  
│   └── 04_compatibility_and_product_report.ipynb  
│  
├── data/  
│   ├── raw/  
│   │   └── product_info.csv  
│   └── processed/  
│       ├── skincare_cleaned.pkl  
│       ├── skincare_final.pkl  
│       └── top_ingredients.pkl  
│  
├── src/  
│   └── app.py  
│
├── results/  
│   ├── models/    
│   │   ├── best_model_final.pkl  
│   │   ├── scaler_final.pkl  
│   │   ├── pair_scores.pkl  
│   │   ├── product_ingredients.pkl  
│   │   ├── product_ratings.pkl  
│   │   ├── product_review_counts.pkl  
│   │   ├── product_names.pkl  
│   │   ├── brand_names.pkl  
│   │   └── review_summaries.pkl  
│   └── figures/  
│       ├── model_comparison.png  
│       └── predicted_vs_actual.png  
│
├── requirements.txt
└── README.md
```
---

---

## How It Works

### 1 — Data Cleaning (Notebook 01)
- Filters the Sephora dataset to 2,420 skincare products
- Cleans and normalizes ingredient lists
- Selects 224 statistically meaningful ingredients (appearing in ≥50 products)

### 2 — Sentiment Analysis (Notebook 02)
- Downloads review data via the Kaggle API
- Scores each review using VADER sentiment analysis
- Aggregates mean sentiment, % positive reviews, and review count per product
- Extracts representative review summaries from the most helpful reviews

### 3 — Model Training (Notebook 03)
- Builds a 227-feature matrix (224 ingredients + 3 sentiment features)
- Trains and compares 3 models:

| Model | RMSE | R² |
|---|---|---|
| Ridge — Ingredients Only (Baseline) | 0.4812 | 0.006 |
| Ridge — Ingredients + Sentiment | 0.3460 | 0.486 |
| **Gradient Boosting — Ingredients + Sentiment** | **0.3372** | **0.512** |

- Gradient Boosting with ingredients and sentiment features achieves the best 
  performance (R² = 0.51), showing that how people talk about a product matters 
  as much as what is in it

### 4 — Compatibility Scoring (Notebook 04)
- Splits products into high-rated (≥ median) and low-rated (< median) groups
- Counts how often every ingredient pair co-occurs in each group
- Computes a compatibility ratio for each pair:
  - Ratio > 1.2 → Low Risk ✅
  - Ratio 0.9–1.2 → Moderate Risk ⚠️
  - Ratio < 0.9 → High Risk ❌
- When a user selects a product, only cross-pairs are scored
  (routine ingredient × new product ingredient)

---

## Getting Started

### 1 — Clone the repo
```bash
git clone https://github.com/AishahAbdulHakeem/skincare-ai.git
cd skincare-ai
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Download the raw data
The raw review files are too large for GitHub. Download them from Kaggle:

[Sephora Products and Skincare Reviews](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

Place `product_info.csv` in `data/raw/`.

### 4 — Run the app
All processed pkl files are included in the repo so you can run the app 
immediately without re-running the notebooks:

```bash
streamlit run src/app.py
```

### 5 — Regenerate from scratch (optional)
If you want to re-run the full pipeline, run the notebooks in order in 
Google Colab. Each notebook saves its outputs to Google Drive and the 
next notebook picks them up automatically.

---

## Dataset

**Sephora Products and Skincare Reviews**
- Source: [Kaggle — nadyinky](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)
- 8,494 products across all categories
- 2,420 skincare products used in this project
- Reviews across multiple CSV files (~1M+ reviews total)

---

## Key Findings

- **Sentiment dominates rating prediction** — mean sentiment score and % 
  positive reviews have F-scores of 2,455 and 1,834 respectively, far higher 
  than any individual ingredient
- **Ingredients still matter** — Gradient Boosting captures ingredient 
  interaction effects that linear models miss, improving R² from 0.49 to 0.51
- **Water normalization matters** — the dataset contains multiple representations 
  of water (water, aqua, water/aqua/eau, etc.) which were standardized to prevent 
  spurious ingredient pairs

---

## Requirements  

- streamlit
- pandas
- numpy
- scikit-learn
- vaderSentiment
- matplotlib 
---

## Author

Built as part of an AI Practicum project.
