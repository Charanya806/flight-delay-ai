# ✈️ Flight Delay Prediction AI

An end-to-end Machine Learning application that predicts whether a flight is likely to be **Delayed** or **On Time** using pre-flight information such as airline, route, scheduled departure time, distance, and flight schedule.

## 🚀 Live Demo

**Streamlit App:** Add your deployed Streamlit Cloud URL here.

## 📌 Project Overview

Flight delays can affect passengers, airlines, airport operations, and overall travel efficiency.

This project uses Machine Learning to predict flight delays **before departure**, helping users identify potential delay risks using information available during flight scheduling.

## 🎯 Objective

The main objective is to build a practical AI-based system that:

* Predicts flight delay risk before departure
* Uses only pre-flight information
* Provides delay probability
* Classifies flights as **Delayed** or **On Time**
* Provides an interactive web interface
* Maintains prediction history
* Displays model evaluation insights

## 🧠 Machine Learning Workflow

```text
Raw Flight Dataset
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Frequency Encoding
        ↓
Model Training
        ↓
Threshold Tuning
        ↓
Model Evaluation
        ↓
Streamlit Deployment
```

## 📊 Features Used

The model uses pre-flight features including:

* Month
* Day
* Day of Week
* Airline
* Origin Airport
* Destination Airport
* Scheduled Departure
* Departure Hour
* Departure Minute
* Distance
* Scheduled Flight Time
* Weekend Indicator
* Route
* Airline + Route
* Distance Category

> **Important:** Actual departure delay information is not used as an input feature, because the goal is to predict the delay before the flight departs.

## 🤖 Model

**HistGradientBoostingClassifier**

The model is trained using engineered flight scheduling and route-related features.

Categorical features are transformed using **frequency encoding**.

A decision threshold of **20%** is used to improve detection of delayed flights.

## 📈 Model Performance

| Metric             | Result |
| ------------------ | -----: |
| ROC-AUC            | 0.7028 |
| Accuracy           | 69.14% |
| Delayed Precision  |    31% |
| Delayed Recall     |    57% |
| Delayed F1 Score   |    40% |
| Decision Threshold |    20% |

The ROC-AUC score indicates that the model has meaningful ability to distinguish between delayed and on-time flights.

## 💻 Tech Stack

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Joblib**
* **Matplotlib**
* **Seaborn**
* **Plotly**
* **Streamlit**
* **Git & GitHub**

## 📂 Project Structure

```text
flight-delay-ai/
│
├── app.py
├── predict.py
├── train_model.py
├── prepare_ml_data.py
├── clean_data.py
├── eda.py
├── evaluation.py
├── feature_importance.py
├── tune_threshold.py
│
├── models/
│   ├── flight_delay_model.pkl
│   ├── frequency_mappings.pkl
│   └── model_features.pkl
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
│
├── airlines.csv
├── airports.csv
├── requirements.txt
├── runtime.txt
└── README.md
```

## ▶️ Run Locally

Clone the repository:

```bash
git clone https://github.com/Charanya806/flight-delay-ai.git
cd flight-delay-ai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🌐 Deployment

The application is deployed using **Streamlit Community Cloud** and connected directly to the GitHub repository.

Every update pushed to the main branch can be deployed to the live application.

## 🔍 Key Insights

Feature importance analysis showed that the most influential features include:

1. Scheduled Departure
2. Month
3. Day
4. Airline Pattern
5. Day of Week

This highlights the importance of **time-based and airline-related patterns** in flight delay prediction.

## ✨ Application Features

* ✈️ Interactive flight prediction
* 📊 Delay probability
* 🟢 On-time / 🔴 Delayed classification
* 📈 Model performance dashboard
* 📋 Prediction history
* 📥 CSV export
* 📊 Confusion matrix
* 📈 ROC curve
* 📉 Precision-Recall curve
* 🎨 Responsive Streamlit UI

## 🔮 Future Improvements

* Real-time weather integration
* Live flight status integration
* Airport congestion data
* Aircraft information
* Real-time airline operational data
* Advanced ensemble models
* Mobile application
* Real-time monitoring dashboard

## 👩‍💻 Author

**Gutti Charanya**

B.Tech – Artificial Intelligence & Data Science
Interested in Machine Learning, AI, Data Science and Software Development.

---

⭐ If you find this project useful, consider giving the repository a star!
