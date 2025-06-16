
# 🧠 Football Match Outcome Prediction  Using Machine Learning and Deep Learning

This project focuses on predicting the outcome of football (soccer) matches using multiple machine learning and deep learning models. The dataset used is based on English Premier League match results and statistics. The main objective is to classify match results as **Home Win**, **Draw**, or **Away Win**.

---

## 📁 Project Structure

- `FINAL_AI.py` – The main implementation script covering data preprocessing, model training, evaluation, and comparison.
- `results.csv` – Dataset used for model training and evaluation.

---

## 🧪 Models Implemented

The following models were developed, trained, and evaluated:

1. **Random Forest Classifier** (with and without hyperparameter tuning)
2. **Support Vector Machine (SVM)** (with and without hyperparameter tuning)
3. **Gradient Boosting Classifier** (with and without hyperparameter tuning)
4. **Feedforward Neural Network (FNN)**
5. **Long Short-Term Memory (LSTM)**

---

## 📊 Features Used

The dataset includes historical match statistics such as:

- Half-time and full-time goals (HTHG, FTAG, etc.)
- Shots, fouls, corners, and cards
- Encoded team identities (HomeTeamID, AwayTeamID)
- Derived features: recent performance, shot-to-goal ratios, team goal efficiency, etc.

---

## 🧹 Data Preprocessing

Key steps:

- Missing value handling with mean/mode imputation
- Feature engineering (e.g., recent performance trends, goal ratios)
- Label encoding for match results (H/D/A ➝ 1/0/2)
- Feature scaling with MinMaxScaler
- Dimensionality reduction and feature selection through correlation analysis

---

## 📈 Evaluation Metrics

Models were evaluated using:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**
- **Confusion Matrices**
- **Comparative Bar Charts** for all metrics

---

## 🧠 Deep Learning Approach

The LSTM model was used to capture temporal trends, taking into account sequence-aware data reshaping for training and testing. It includes:

- Two stacked LSTM layers
- Dropout layers for regularization
- Dense output layer with softmax activation

---

## 🛠 Tools & Libraries

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- TensorFlow / Keras

---

## 📌 Results

All models performed well with varying degrees of precision. Hyperparameter tuning improved most models’ accuracy. Visualizations and performance metrics were used to compare and interpret model performance clearly.

---

## 🚀 How to Run

1. Ensure `results.csv` is in the same directory as the script.
2. Run `FINAL_AI.py` in a Jupyter Notebook or compatible Python environment.
3. View performance metrics and visual comparisons in the output.

---

## 📄 License

This project is provided for educational and research purposes.

---
