# Machine Learning Assignment 2: Email Spam Classification System

**Course:** M.Tech (AIML) - Machine Learning (S1-26_AIMLZG565)  
**Institution:** BITS Pilani (Work Integrated Learning Programmes Division)  
**Student ID:** 2026AK05158  

---

## a. Problem Statement

Unsolicited bulk email, commonly known as spam, creates significant operational bottlenecks, security risks (such as phishing and malware distribution), and storage overhead for individuals and corporate networks. The objective of this assignment is to design, train, evaluate, and deploy an end-to-end machine learning classification system capable of accurately discriminating between legitimate emails ("ham") and unsolicited messages ("spam").

The problem is framed as a supervised binary classification task:
$$\hat{y} \in \{0, 1\}$$
where $y = 0$ corresponds to non-spam (legitimate correspondence) and $y = 1$ denotes spam. The evaluation prioritizes both overall discrimination capability (high ROC-AUC and MCC) and a balanced trade-off between Precision and Recall. In practical spam filtering, misclassifying a critical legitimate email as spam (False Positive) can have severe consequences, while letting spam reach the inbox (False Negative) degrades user experience and trust.

---

## b. Dataset Description

The dataset selected for this study is the benchmark **UCI Spambase Dataset** (contributed by Mark Hopkins, Erik Reeber, George Forman, and Jaap Suermondt of Hewlett-Packard Laboratories, available through the UCI Machine Learning Repository and OpenML).

### Dataset Attributes and Structure:
- **Total Instances:** 4,601 email records
  - Legitimate emails (class = 0): 2,788 instances (60.60%)
  - Spam emails (class = 1): 1,813 instances (39.40%)
- **Total Features:** 57 continuous numeric predictor attributes (exceeding the minimum requirement of 12 features):
  1. **Word Frequency Attributes (48 continuous features):** Measure the percentage of words in the email body that match a specific token (e.g., `word_freq_make`, `word_freq_address`, `word_freq_all`, `word_freq_free`, `word_freq_business`, `word_freq_email`, `word_freq_money`, `word_freq_credit`, `word_freq_hp`, `word_freq_george`). The values range from 0.0 to 100.0%.
  2. **Character Frequency Attributes (6 continuous features):** Measure the percentage of characters in the message matching specific special punctuation marks frequently leveraged in marketing or phishing (e.g., `char_freq_semicolon`, `char_freq_round_bracket`, `char_freq_square_bracket`, `char_freq_exclamation`, `char_freq_dollar`, `char_freq_hash`).
  3. **Capital Run-Length Attributes (3 continuous features):** Measure patterns of consecutive uppercase letters:
     - `capital_run_length_average`: Average length of uninterrupted sequences of capital letters.
     - `capital_run_length_longest`: Maximum length of an uninterrupted sequence of capital letters.
     - `capital_run_length_total`: Total count of capital letters across the entire email.
- **Target Variable:** `class` (Integer: 0 for non-spam, 1 for spam).
- **Missing Values:** None. All 57 attributes are populated with valid floating-point values.

### Data Partitioning:
The full dataset of 4,601 instances was partitioned using a stratified 80/20 train/test split (`random_state=42`) to preserve the 60.6% / 39.4% class ratio across both partitions:
- **Training Subset:** 3,680 instances (used to fit models and compute feature scaling parameters).
- **Hold-out Test Subset:** 921 instances (exported as `test_data.csv` with all 57 feature columns and the `class` label).

---

## c. GitHub Repository Link

- **GitHub Repository:** `https://github.com/lokeshpanda/email-spam-classifier-bits`
- **Streamlit Community Cloud Live App:** `https://email-spam-classifier-bits.streamlit.app`

*(Note: Verify the exact URL depending on your personal repository configuration.)*

---

## d. Models Used & Evaluation Metrics

Five distinct classification algorithms representing different inductive biases were implemented and evaluated on the same 921 hold-out test samples:
1. **Logistic Regression:** L2-regularized linear model fitted using the `lbfgs` solver (`max_iter=1000`). Features standardized via `StandardScaler`.
2. **Decision Tree Classifier:** Non-parametric recursive splitting model regularized with `max_depth=6` to prevent overfitting on long-tail word frequencies. Evaluated on raw feature space.
3. **K-Nearest Neighbor (KNN):** Distance-based instance learning using Euclidean metric with $k=5$ neighbors. Evaluated on standardized feature space.
4. **Naive Bayes (GaussianNB):** Probabilistic classifier assuming conditional independence and Gaussian class-conditional distributions. Evaluated on raw feature space.
5. **Support Vector Machine (SVM):** Maximum-margin separator with a Radial Basis Function (RBF) kernel and calibrated probability estimates. Evaluated on standardized feature space.

### Comparison Table

The evaluation metrics calculated across all 5 models on the identical test set (`test_data.csv`, $N=921$) are summarized below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **0.9294** | **0.9702** | 0.9209 | 0.8981 | **0.9093** | **0.8518** |
| **Decision Tree** | 0.9034 | 0.9266 | **0.9308** | 0.8154 | 0.8693 | 0.7976 |
| **KNN** | 0.9077 | 0.9506 | 0.8861 | 0.8788 | 0.8824 | 0.8065 |
| **Naive Bayes** | 0.8339 | 0.9449 | 0.7178 | **0.9532** | 0.8189 | 0.6941 |
| **Support Vector Machine** | 0.9273 | 0.9667 | 0.9277 | 0.8843 | 0.9055 | 0.8471 |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Delivered the strongest overall performance across the board with the highest Accuracy (0.9294), highest AUC (0.9702), highest F1-Score (0.9093), and highest MCC (0.8518). Because the Spambase dataset has 57 features that contribute additive evidence towards spam likelihood (e.g. positive log-odds weights for currency tokens like `char_freq_dollar` and negative weights for personal domain tokens like `word_freq_hp`), standardizing the inputs and fitting a linear decision boundary with L2 regularization generalized exceptionally well without overfitting. |
| **Decision Tree** | Achieved the highest Precision (0.9308) among all models, misclassifying only 22 legitimate emails as spam. By constraining the tree depth to `max_depth=6`, the model identified clean orthogonal threshold rules on high-impact discriminators (specifically `char_freq_dollar` and `capital_run_length_longest`). However, its Recall dropped to 0.8154 (67 spam emails missed), as rigid axis-parallel splits struggle to capture subtle combinations of lower-frequency keywords. |
| **KNN** | Demonstrated solid, balanced classification (Accuracy 0.9077, AUC 0.9506, Precision 0.8861, Recall 0.8788, MCC 0.8065). With standardized Euclidean distances, query emails reliably mapped near homogeneous clusters in the 57-dimensional metric space. Its primary limitation is runtime compute overhead during prediction and sensitivity to local density variations compared to parametric baselines. |
| **Naive Bayes** | Produced the highest Recall of any model (0.9532), capturing 346 out of 363 spam emails (only 17 missed). However, it suffered the lowest Precision (0.7178) and lowest Accuracy (0.8339), generating 136 False Positives. This behavior directly stems from its class-conditional independence assumption: heavily correlated spam markers (such as multiple financial terms and exclamation marks appearing together) compound probabilities unrealistically, causing borderline legitimate emails to be flagged as spam. |
| **Support Vector Machine** | Delivered the second-best overall performance (Accuracy 0.9273, AUC 0.9667, Precision 0.9277, Recall 0.8843, F1 0.9055, MCC 0.8471). The non-linear RBF kernel mapped the 57 standardized continuous features into a high-dimensional Hilbert space, constructing an effective maximum-margin boundary that is robust against noise and outliers. Its precision-recall balance is virtually neck-and-neck with Logistic Regression. |
| **Overall, Winner for your dataset?** | **Logistic Regression is the overall winner for this dataset.** It achieves the highest Accuracy (92.94%), highest ROC-AUC (0.9702), highest F1-Score (0.9093), and highest Matthews Correlation Coefficient (0.8518), while maintaining a very low False Positive rate (only 28 false alarms out of 558 legitimate emails). In addition to superior statistical metrics, Logistic Regression offers fast inference latency, minimal memory footprint (1.3 KB serialized model size), and straightforward coefficient interpretability, making it the most practical and dependable choice for production email filtering. |

---

## Repository Structure

```
.
├── app.py                     # Interactive Streamlit web application
├── train.py                   # Complete model training and evaluation script
├── requirements.txt           # Python dependencies for local and cloud deployment
├── README.md                  # Comprehensive documentation and evaluation report
├── test_data.csv              # Hold-out test dataset (921 rows, 57 features + 1 target)
├── evaluation_results.csv     # Tabular metric outputs across all five models
├── ML_Assignment_2_AIML_26Oct26.pdf # Official assignment brief
└── model/                     # Serialized pre-trained artifacts
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── logistic_regression.joblib
    ├── naive_bayes.joblib
    ├── scaler.joblib
    └── svm.joblib
```

---

## Instructions to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lokeshpanda/email-spam-classifier-bits.git
   cd email-spam-classifier-bits
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Re-run model training:**
   ```bash
   python train.py
   ```

4. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```

5. **Interact with the App:**
   - Choose a model from the sidebar dropdown.
   - Inspect Accuracy, AUC, Precision, Recall, F1, and MCC metrics computed in real-time.
   - Examine the interactive Confusion Matrix heatmap and Classification Report.
   - Test custom data by uploading `test_data.csv` via the sidebar uploader.
