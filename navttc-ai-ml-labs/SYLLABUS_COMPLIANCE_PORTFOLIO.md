# NAVTTC AI/ML Program — Syllabus Compliance Portfolio

**Course:** Artificial Intelligence (Machine Learning & Deep Learning)  
**Duration:** 12 Weeks | 4 hrs/day | 80% Practical  
**Repository:** `navttc-ai-ml-labs`  
**Alignment:** Official NAVTTC Lesson Plan + Annexure-I (Tasks 1–65) + Global Industry Standards

---

## Executive Summary

This portfolio documents **complete, evidence-based coverage** of the NAVTTC 3-month AI/ML curriculum. Work is organized as runnable labs with outputs, mapped week-by-week to the official syllabus, and cross-referenced to global competency frameworks used by employers (NetSol, Arbisoft, Afiniti, Microsoft Azure certification paths).

| Metric | Status |
|--------|--------|
| Official Tasks (Annexure-I) | **65 / 65** mapped |
| Weekly Modules (Week 1–12) | **12 / 12** addressed |
| Core Libraries (NumPy, Pandas, scikit-learn, TensorFlow/Keras) | Covered |
| PyTorch (syllabus requirement) | `lab72_pytorch_fundamentals.py` |
| NLP Specialization (POS, NER, embeddings, deployment) | Labs 60–62, 70, 73 |
| Deep Learning (CNN, RNN, LSTM, sequence models) | Labs 56–64, 71 |
| Microsoft Azure AI (Task 65 + AI-900 alignment) | `lab65_azure_ai_overview.py` |
| Capstone / Employable Project | `lab74_capstone_sentiment_api.py` |

---

## Global Stakeholder Alignment

### Government & NAVTTC Objectives
- **PM Hunarmand Pakistan / Skills for All** — hands-on, employable skills
- **80% practical / 20% theory** — every topic backed by executable code
- **Job roles targeted:** AI Associate Engineer, ML Analyst, Assistant Data Analyst, Research Assistant
- **Industry partners cited in syllabus:** NetSol, Arbisoft, Careem, Afiniti, Confiz, Crossover

### International Industry Standards
| Framework | How This Portfolio Aligns |
|-----------|----------------------------|
| **Microsoft AI-900** (Azure AI Fundamentals) | Lab 65 — Cognitive Services, Azure ML, OpenAI, Responsible AI |
| **scikit-learn ML Pipeline** | Labs 51–55, 68–69 — EDA → train → evaluate → persist |
| **TensorFlow/Keras** | Labs 56–64, 71 — NN, MLP, CNN, LSTM |
| **PyTorch** | Lab 72 — tensors, autograd, simple classifier |
| **NLP Production Stack** | Labs 60–62, 70, 73 — NLTK, embeddings, model save/load |
| **MLOps Basics** | Lab 73 — serialization, inference script, deployment pattern |

### What Reviewers Should Look For
1. Open `navttc_practice_workbook.ipynb` — Labs 2–55 (Python → ML)
2. Run any `labXX_*.py` file — each maps to syllabus week
3. Check plot outputs (`labXX_*.png`) after running scripts
4. Read this file for week-by-week traceability

---

## 12-Week Coverage Matrix

| Week | Syllabus Module | Evidence | Status |
|------|-----------------|----------|--------|
| **1** | Linux, Python basics, data structures, control flow | Workbook Labs 2–25 | ✅ Complete |
| **1** | Task 1: Environment setup (Anaconda, TF, PyTorch, VS Code) | `lab01_environment_setup.md` | ✅ Complete |
| **2** | OOP, file handling, lambda/map/filter, statistics intro | Workbook Labs 26–27, 49–50; `lab66_python_oop_files.py` | ✅ Complete |
| **3** | Descriptive stats, probability, NumPy deep dive | Workbook Labs 28–40; `lab67_seaborn_probability_eda.py` | ✅ Complete |
| **4** | Pandas, Seaborn, ML pipeline, gradient descent | Workbook Labs 41–48; `lab68_gradient_descent_polynomial.py` | ✅ Complete |
| **5** | Linear/logistic regression, polynomial regression | Workbook Labs 51–52; Lab 68 | ✅ Complete |
| **6** | NLP, SVM, decision trees, evaluation metrics, Random Forest | Labs 53–55, 60–62; `lab69_ensemble_ml.py`, `lab70_nlp_ner_embeddings.py` | ✅ Complete |
| **7** | Neural networks, MLP, feedforward, loss functions, boosting | Labs 56–59; Lab 69 | ✅ Complete |
| **8** | CNN, 1D CNN for text, RNN, LSTM, GRU | Labs 63–64; `lab71_lstm_text_classifier.py` | ✅ Complete |
| **9** | Word embeddings, Word2Vec, sequence models | `lab70_nlp_ner_embeddings.py`, Lab 71 | ✅ Complete |
| **10** | Bi-LSTM, attention concepts, employable project | Lab 71, `lab74_capstone_sentiment_api.py` | ✅ Complete |
| **11–12** | Microsoft Azure AI services, deployment, project viva | Lab 65, Lab 73, Lab 74 | ✅ Complete |

---

## Annexure-I Task Mapping (Tasks 1–65)

| Task | Title | Evidence File | Week |
|------|-------|---------------|------|
| 1 | Installation | `lab01_environment_setup.md` | 1 |
| 2 | Linux Commands | Workbook — Week 1 Linux reference | 1 |
| 3–25 | Python fundamentals | `navttc_practice_workbook.ipynb` | 1 |
| 26–27 | Exception handling | Workbook Labs 26–27 | 2 |
| 28–40 | NumPy | Workbook Labs 28–40 | 3–4 |
| 41–48 | Pandas | Workbook Labs 41–48 | 3–4 |
| 49–50 | Statistics | Workbook Labs 49–50 | 2 |
| 51 | Linear Regression | Workbook Lab 51 | 5 |
| 52 | Logistic Regression | Workbook Lab 52 | 5 |
| 53 | Decision Tree | Workbook Lab 53 | 6 |
| 54 | SVM | Workbook Lab 54 | 6 |
| 55 | Time Series | Workbook Lab 55 | 6 |
| 56 | Neural Networks | `lab56_neural_network_basics.py` | 7 |
| 57 | MLP | `lab57_mlp_mnist.py` | 7 |
| 58 | Feedforward NN | `lab58_feedforward_classifier.py` | 7 |
| 59 | Loss Functions | `lab59_loss_functions.py` | 7 |
| 60 | NLP Linguistics | `lab60_62_nlp_pipeline.py` | 6–7 |
| 61 | Text Processing | `lab60_62_nlp_pipeline.py` | 6–7 |
| 62 | Text Analysis | `lab60_62_nlp_pipeline.py` | 6–7 |
| 63 | CNN Demo | `lab63_cnn_mnist.py` | 7–8 |
| 64 | CNN CIFAR-10 | `lab64_cnn_cifar10.py` | 7–8 |
| 65 | Microsoft Azure | `lab65_azure_ai_overview.py` | 11–12 |

### Supplementary Labs (Syllabus Topics Beyond Annexure Tasks)

| Lab | Syllabus Topic | File |
|-----|----------------|------|
| 66 | OOP, inheritance, file I/O | `lab66_python_oop_files.py` |
| 67 | Seaborn EDA, probability plots | `lab67_seaborn_probability_eda.py` |
| 68 | Gradient descent, polynomial regression | `lab68_gradient_descent_polynomial.py` |
| 69 | Random Forest, XGBoost (bagging/boosting) | `lab69_ensemble_ml.py` |
| 70 | NER, BoW, TF-IDF, Word2Vec | `lab70_nlp_ner_embeddings.py` |
| 71 | LSTM sequence classifier (text) | `lab71_lstm_text_classifier.py` |
| 72 | PyTorch fundamentals | `lab72_pytorch_fundamentals.py` |
| 73 | Model persistence & deployment | `lab73_model_deployment.py` |
| 74 | Capstone: end-to-end sentiment pipeline | `lab74_capstone_sentiment_api.py` |

---

## Competency Checklist (Learning Outcomes)

| Competency (from official syllabus) | Demonstrated In |
|-------------------------------------|-----------------|
| Core AI/ML concepts | Workbook + Labs 51–59 |
| State-of-the-art ML techniques | Labs 51–55, 69 |
| Exploratory data analysis | Workbook 41–48, Lab 67 |
| Model design & evaluation | All ML/DL labs |
| scikit-learn, pandas, numpy | Workbook 28–55 |
| TensorFlow, Keras | Labs 56–64, 71 |
| PyTorch | Lab 72 |
| NLP: tokenization, POS, NER | Labs 60–62, 70 |
| Word embeddings | Lab 70 |
| Deploy NLP model | Labs 73, 74 |
| Azure AI APIs & services | Lab 65 |
| CNN for images | Labs 63–64 |
| RNN/LSTM for sequences | Lab 71 |
| Professional project delivery | Lab 74 |

---

## Technology Stack Summary

```
Python 3.x
├── Data: NumPy, Pandas, Seaborn
├── ML: scikit-learn, XGBoost
├── DL: TensorFlow/Keras, PyTorch
├── NLP: NLTK, Gensim (Word2Vec)
├── Viz: Matplotlib, Seaborn
└── Cloud: Microsoft Azure AI (documented + SDK patterns)
```

---

## How to Verify (For Instructor / Stakeholder)

```bash
# 1. Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost tensorflow torch nltk gensim

# 2. Run workbook (Jupyter)
jupyter notebook navttc_practice_workbook.ipynb

# 3. Run any lab
python lab56_neural_network_basics.py
python lab74_capstone_sentiment_api.py

# 4. Check generated artifacts
#    - labXX_*.png plots
#    - saved_models/ folder (Lab 73–74)
```

---

## Capstone Project Summary (Lab 74)

**Title:** Product Review Sentiment Analyzer  
**Problem:** Classify customer reviews as positive/negative for e-commerce decision support  
**Pipeline:** Data load → text cleaning → TF-IDF/features → train (Logistic + LSTM) → evaluate → save model → inference demo  
**Business relevance:** Used by Careem, Daraz, and SaaS product teams for feedback analysis  
**NAVTTC Week 10–12 alignment:** Employable project with viva-ready documentation

---

## Viva Preparation — Key Talking Points

1. **Linux & Python (Week 1):** Explain `pwd`, `chmod`, loops, data structures with workbook examples.
2. **EDA (Week 3–4):** Walk through Pandas cleaning on `learner_scores.csv` and Seaborn heatmap in Lab 67.
3. **ML (Week 5–6):** Compare linear vs logistic regression; explain SVM kernel idea; show decision tree plot.
4. **DL (Week 7–8):** Draw CNN architecture (Conv → Pool → Dense); explain why XOR needs hidden layer.
5. **NLP (Week 6, 9):** Demo stemming vs lemmatization; explain Word2Vec intuition in Lab 70.
6. **Azure (Week 11):** Name 3 Cognitive Services (Vision, Language, Speech) and one use case each.
7. **Capstone (Week 10–12):** Present end-to-end flow from Lab 74 with accuracy metrics.

---

## References

- Official NAVTTC Lesson Plan: `Navtaccoursesyllabus.md`
- Microsoft Learn: [Azure AI Fundamentals](https://learn.microsoft.com/en-us/training/paths/get-started-with-artificial-intelligence-on-azure/)
- Annexure-I task URLs (W3Schools, GeeksforGeeks, Analytics Vidhya, etc.)

---

*This portfolio is designed for NAVTTC assessment, employer review, and GitHub evidence of complete 12-week practical work.*
