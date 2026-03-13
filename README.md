# Fingerprint Microbiome Classification

This project investigates whether DNA traces left on a keyboard can be used to determine if a user typed with their **left hand or right hand**.

Recent studies show that microbial DNA patterns left by fingerprints can provide unique biological signatures. In this project, machine learning techniques are used to classify whether the DNA sample originates from the **left hand or right hand** of a user.

This problem is treated as a **binary classification task**.

---

## Project Background

This project was developed during my **Bachelor's degree in Computer Engineering at Erciyes University** as part of the **Pattern Recognition** course.

The goal was to apply machine learning techniques to analyze biological fingerprint microbiome data.

Currently, I am pursuing an **M.Sc. in Computer Science at HAW Kiel University of Applied Sciences (Germany)**.

---

## Dataset

The dataset contains microbiome DNA features extracted from fingerprint traces collected on computer keyboards.

Dataset characteristics:

| Property | Value |
|--------|--------|
| Total samples | 271 |
| Features per sample | 3302 |
| Classes | Left hand / Right hand |

Class distribution:

| Class | Samples |
|------|--------|
| Left hand | 136 |
| Right hand | 135 |

Dataset location:

data/otu.csv

Each row represents microbial DNA features extracted from a fingerprint sample.

---

## Machine Learning Pipeline

The following steps were applied:

1. Data loading using **Pandas**
2. Label encoding for class labels
3. Dataset splitting using **train_test_split**
4. Feature normalization using **StandardScaler**
5. Model training using **MLPClassifier**
6. Performance evaluation using classification metrics

---

## Model

The classification algorithm used in this project is:

**Multi-Layer Perceptron (MLP)**

Library used:

scikit-learn

MLP was chosen because it performs well on high-dimensional datasets such as microbiome feature vectors.

---

## Performance Evaluation

The model was evaluated using the following metrics:

- Accuracy  
- Sensitivity  
- Specificity  
- ROC AUC  
- Confusion Matrix  

Results:

Accuracy: **0.73**

Sensitivity: **0.73**

Specificity: **0.73**

ROC AUC: **0.727**

These results indicate that the model achieves balanced classification performance between the two classes.

---

## Project Structure

PatternTermProject  
│  
├── src  
│   └── main.py  
│  
├── data  
│   └── otu.csv  
│  
├── report  
│   └── PR_midterm_project.pdf  
│  
├── Report[ENG].txt  
├── Rapor[TR].txt  
└── README.md  

---

## Technologies Used

- Python  
- Pandas  
- NumPy  
- Scikit-learn  

---

## Author

**Furkan Yilmaz**

Developed during my **Bachelor's degree in Computer Engineering at Erciyes University**.

Currently pursuing an **M.Sc. in Computer Science at HAW Kiel University of Applied Sciences (Germany)**.

