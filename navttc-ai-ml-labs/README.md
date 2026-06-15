# NAVTTC AI/ML Labs — Full-Stack Machine Learning Portfolio

> **Government-certified 12-week AI curriculum · 65 hands-on labs · Python to Deep Learning to Azure AI**  
> Runnable code, syllabus-mapped evidence, and a production-style capstone — built for instructors, HR, and tech reviewers.

[![NAVTTC](https://img.shields.io/badge/Program-NAVTTC%20AI%2FML-blue)](Navtaccoursesyllabus.md)
[![Duration](https://img.shields.io/badge/Duration-12%20Weeks-green)](Navtaccoursesyllabus.md)
[![Tasks](https://img.shields.io/badge/Official%20Tasks-65%2F65-brightgreen)](SYLLABUS_COMPLIANCE_PORTFOLIO.md)
[![Practical](https://img.shields.io/badge/Lab%20Work-80%25%20Hands--on-orange)](Navtaccoursesyllabus.md)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](lab01_environment_setup.md)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)](lab56_neural_network_basics.py)
[![PyTorch](https://img.shields.io/badge/PyTorch-Fundamentals-EE4C2C?logo=pytorch&logoColor=white)](lab72_pytorch_fundamentals.py)
[![Azure](https://img.shields.io/badge/Microsoft-Azure%20AI-0078D4?logo=microsoftazure&logoColor=white)](lab65_azure_ai_overview.py)

---

## Who This Repo Is For

| Audience | Start Here |
|----------|------------|
| **Instructors / NAVTTC assessors** | [SYLLABUS_COMPLIANCE_PORTFOLIO.md](SYLLABUS_COMPLIANCE_PORTFOLIO.md) — week-by-week proof, Tasks 1–65 map |
| **HR & hiring managers** | [Capstone project](#capstone-project) + [Skills demonstrated](#skills-demonstrated) |
| **Developers & reviewers** | [navttc_practice_workbook.ipynb](navttc_practice_workbook.ipynb) + `lab*.py` scripts |
| **Students** | [Setup](#quick-start) → run labs in order |

---

## Overview

This repository is a **complete practical portfolio** for Pakistan's **NAVTTC Artificial Intelligence (Machine Learning & Deep Learning)** program under the Prime Minister's *Hunarmand Pakistan / Skills for All* initiative.

It is not slide-deck theory — it is **executable code** across the full 3-month syllabus: Linux and Python foundations, exploratory data analysis, classical machine learning, natural language processing, neural networks, CNNs, LSTMs, PyTorch, model deployment, Microsoft Azure AI, and an employable capstone project.

**Every official Annexure-I task (1–65) is mapped to a file.** Supplementary labs (66–74) cover advanced syllabus topics including ensemble methods, Word2Vec, gradient descent, and end-to-end deployment.

---

## Highlights at a Glance

| Metric | Coverage |
|--------|----------|
| Program duration | 12 weeks · 4 hrs/day |
| Official NAVTTC tasks | **65 / 65** documented |
| Weekly modules | **12 / 12** |
| Jupyter workbook labs | Labs 2–55 |
| Deep learning scripts | Labs 56–74 |
| Capstone | Product Review Sentiment Analyzer |
| Industry alignment | Azure AI-900 · scikit-learn pipeline · TF/Keras · PyTorch |

---

## Skills Demonstrated

**Programming & Foundations**  
Linux CLI · Python · OOP · file I/O · exception handling · data structures

**Data Science**  
NumPy · Pandas · Seaborn · EDA · correlation · probability · time series

**Machine Learning**  
Linear & logistic regression · decision trees · SVM · Random Forest · XGBoost · gradient descent · model evaluation

**Deep Learning**  
Neural networks · MLP · CNN (MNIST, CIFAR-10) · LSTM · loss functions · TensorFlow/Keras · PyTorch

**NLP**  
Tokenization · POS tagging · NER · sentiment analysis · BoW · TF-IDF · Word2Vec

**Cloud & Deployment**  
Microsoft Azure AI services · model persistence · inference pipeline · REST deployment pattern

---

## Capstone Project

**Lab 74 — Product Review Sentiment Analyzer**

End-to-end ML pipeline for e-commerce review classification:

`Data → text features (TF-IDF) → train → evaluate → save model → live inference`

Relevant for product teams, customer support automation, and feedback analytics — aligned with NAVTCC Week 10–12 employable project requirements.

```bash
python lab74_capstone_sentiment_api.py
```

---

## Repository Structure

```
navttc-ai-ml-labs/
│
├── README.md                          ← You are here
├── SYLLABUS_COMPLIANCE_PORTFOLIO.md   ← Full syllabus evidence (for assessors)
├── Navtaccoursesyllabus.md            ← Official NAVTTC lesson plan
│
├── navttc_practice_workbook.ipynb     ← Labs 2–55 (Python → ML)
├── learner_scores.csv                 ← Sample dataset
├── lab01_environment_setup.md         ← Task 1: toolchain setup
│
├── lab56–lab65                        ← Official tasks (NN, CNN, NLP, Azure)
└── lab66–lab74                        ← OOP, Seaborn, ensembles, LSTM, PyTorch, capstone
```

---

## 12-Week Learning Path

| Week | Focus | Key Artifacts |
|------|-------|---------------|
| 1 | Linux · Python basics | Workbook · `lab01` |
| 2 | OOP · files · statistics | Workbook · `lab66` |
| 3–4 | NumPy · Pandas · Seaborn | Workbook · `lab67` |
| 4–5 | ML pipeline · gradient descent | Workbook · `lab68` |
| 5–6 | Regression · SVM · trees · NLP | Workbook · `lab69`–`lab70` |
| 7–8 | Neural nets · CNN · LSTM | `lab56`–`lab64` · `lab71` |
| 9 | Embeddings · PyTorch | `lab70` · `lab72` |
| 10–12 | Capstone · Azure · deployment | `lab65` · `lab73` · `lab74` |

---

## Tech Stack

```
Python 3.x
├── Data       → NumPy, Pandas, Seaborn, Matplotlib
├── ML         → scikit-learn, XGBoost
├── DL         → TensorFlow, Keras, PyTorch
├── NLP        → NLTK, Gensim
└── Cloud      → Microsoft Azure AI (documented + SDK patterns)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow torch nltk gensim joblib jupyter
```

### 2. Run the workbook

```bash
jupyter notebook navttc_practice_workbook.ipynb
```

### 3. Run deep learning & capstone labs

```bash
python lab56_neural_network_basics.py
python lab71_lstm_text_classifier.py
python lab74_capstone_sentiment_api.py
```

### 4. Verify full syllabus coverage

Open **[SYLLABUS_COMPLIANCE_PORTFOLIO.md](SYLLABUS_COMPLIANCE_PORTFOLIO.md)** for the complete task map and reviewer checklist.

---

## Career Paths This Portfolio Supports

Aligned with NAVTTC syllabus job outcomes and hiring partners (NetSol, Arbisoft, Careem, Afiniti, Confiz, and others):

- AI Associate Engineer  
- Machine Learning Analyst  
- Assistant Data Analyst  
- Deep Learning / NLP Junior Developer  
- Research Assistant  

---

## For Reviewers — 60-Second Verification

1. Open `SYLLABUS_COMPLIANCE_PORTFOLIO.md` → confirm **65/65 tasks**  
2. Skim `navttc_practice_workbook.ipynb` → Labs 2–55  
3. Run `python lab74_capstone_sentiment_api.py` → check output + `lab74_capstone_results.png`  
4. Browse `lab65_azure_ai_overview.py` → Azure AI competency  

---

## License & Attribution

Educational portfolio developed as part of the **NAVTTC AI/ML training program**.  
Syllabus reference: Government of Pakistan, National Vocational and Technical Training Commission (NAVTTC).

---

**Questions for viva or technical review?** Every lab maps to a syllabus week — see the compliance portfolio for traceability.
