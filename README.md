# 🧴 Ingredient-Level Skincare Compatibility & Rating Prediction

## Team Members
- Aishah Abdul-Hakeemn (aaa279)
- Siri Reddy (smr355)
- Jaydon Akintlotan (jaa374)

---

## Project Overview

This project builds an AI system that predicts the average rating of a skincare product using ingredient features and review sentiment, while also estimating compatibility with a user’s current skincare routine.

**Inputs:**
- A Sephora skincare product  
- A user-selected list of current skincare ingredients  

**Outputs:**
- Predicted average rating (regression)
- Compatibility risk level (Low / Moderate / High)
- Structured explanation of compatibility factors  

The goal is to evaluate whether adding ingredient compatibility features improves predictive performance over a baseline ingredient-only model.

---

## AI Components

- Supervised Learning (Regression)
- Natural Language Processing (Review Sentiment Aggregation)
- Statistical Co-occurrence Modeling
- Feature Engineering

---

## Dataset

We use the Sephora Products and Reviews dataset from Kaggle, including:

- Product metadata (ingredients, category, ratings)
- Individual product reviews
- Review text and ratings

Only products in the **"Skincare"** category are used. 

[Kaggle Dataset](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)
---

## Methodology Overview

1. Clean and standardize ingredient lists  
2. Select top N most frequent ingredients to reduce sparsity  
3. Train baseline regression model using ingredient features  
4. Build compatibility score using ingredient co-occurrence patterns  
5. Add compatibility feature to regression model  
6. Incorporate sentiment features from review text  
7. Compare model performance across feature sets  


---

## Evaluation Metrics

We evaluate model performance using:

- Root Mean Squared Error (RMSE)
- R² Score

We compare:

- Ingredients-only baseline  
- Ingredients + compatibility  
- Ingredients + compatibility + sentiment  

---

## Current Status

- Repository initialized  
- Dataset uploaded  
- Ingredient cleaning and frequency analysis in progress  

---

## Running the Project (To Be Completed)


