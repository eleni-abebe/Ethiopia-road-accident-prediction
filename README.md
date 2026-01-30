🚦 Ethiopia Road Accident Severity Prediction

A machine learning–based system for predicting road accident severity in Addis Ababa, Ethiopia, using historical road traffic accident data.
The project includes data preprocessing, model training, evaluation, and an interactive Streamlit web application.

📌 Project Overview

Road traffic accidents are a major public safety issue in Ethiopia.
This project aims to predict the severity of road accidents (`Fatal`, `Serious`, `Slight`) using machine learning models trained on historical accident records.

The system:

* Cleans and preprocesses real accident data
* Handles severe class imbalance using SMOTE
* Trains and evaluates multiple ML models
* Allows users to interactively test predictions via a Streamlit web app

🗺️ Dataset

* Source:Addis Ababa Road Traffic Accident Dataset
* Country:Ethiopia 🇪🇹
* Target Variable:Accident_severity

  * `0` → Fatal
  * `1` → Serious
  * `2` → Slight

Key Features Used

 Hour of the accident
 Number of vehicles involved
 Number of casualties
 Encoded accident characteristics (cause, road condition, etc.)

🧠 Models Trained

The following models were trained and evaluated:

| Model                   | Description                |
| ----------------------- | -------------------------- |
| Logistic Regression     | Baseline linear classifier |
| Random Forest           | Ensemble tree-based model  |
| Gradient Boosting       | Best performing model      |

📊 Evaluation Metric
🔍 Model Performance (Macro F1)

| Model                 | Macro F1-score |
| --------------------- | -------------- |
| Logistic Regression   | 0.31           |
| Random Forest         | 0.37           |
| **Gradient Boosting** | **0.43** ⭐     |

🧪 Project Structure


ethiopia-road-accident-prediction/
│
├── data/
│   ├── train_X.csv
│   ├── train_y.csv
│   ├── test_X.csv
│   └── test_y.csv
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
│
├── app.py
├── model_lr.pkl
├── model_rf_best.pkl
├── model_gb_best.pkl
├── requirements.txt
└── README.md


⚙️ Installation

1️⃣ Create and activate a virtual environment

```bash
python -m venv venv
source venv/Scripts/activate   # Windows (Git Bash)
```

2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

🏋️ Model Training

Train all models and save them as `.pkl` files:

```bash
python src/train.py
```

 📈 Model Evaluation

Evaluate all trained models on the test set:

```bash
python src/evaluate.py
```
🌐 Streamlit Web App

### Run the application

```bash
streamlit run app.py
```

### App Features

* Choose which model to use:

  * Logistic Regression
  * Random Forest
  * Gradient Boosting (recommended)
* Input accident details
* View predicted severity + probability distribution

⚠️ **Disclaimer:**
The app is a **demo version** and uses only a subset of features.
Most other features are set to default values.

