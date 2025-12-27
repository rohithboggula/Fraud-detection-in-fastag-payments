
---

# FASTag Fraud Detection System

A **machine learning–based web application** built with **Streamlit** to detect fraudulent FASTag toll payment transactions.
The system supports **manual prediction**, **batch CSV uploads**, **model comparison**, and **statistical dashboards** using real-world–inspired NETC data.

---

##  Features

###  Fraud Prediction

* Manual transaction input for real-time fraud prediction
* Batch prediction via CSV upload
* Random Forest–based classification model
* Automatic feature engineering and encoding

###  Analytics & Visualization

* Model performance comparison (Logistic Regression, Decision Tree, Random Forest, SVM, KNN)
* Confusion matrices and accuracy comparison
* Statistical dashboards (fraud trends by day, month, state)
* KDE plots, histograms, regression plots, and correlation heatmaps

###  Real-Time Statistics

* NETC monthly and yearly transaction analysis
* Bank-wise FASTag statistics
* RBI regulatory insights
* Trend and distribution analysis

---

##  Tech Stack

* **Frontend:** Streamlit
* **Backend:** Python
* **ML Libraries:** scikit-learn
* **Data Processing:** pandas, numpy
* **Visualization:** matplotlib, seaborn
* **Model Persistence:** joblib

---

##  Project Structure

```
fastag-fraud-detection/
│
├── app.py                     # Streamlit application
├── model.py                   # ML models and evaluation
├── train_model.py             # Model training & saving
├── stats.py                   # Statistical analysis & plots
├── live_stats.py              # Real-time NETC analysis
│
├── data/
│   ├── cleaned_data.csv
│   ├── FastagFraudDetection.csv
│   ├── NETC_Monthly_Transactions.csv
│   ├── netc_yearly_data.csv
│   └── netc_banks_data.csv
│
├── random_forest_model.joblib
├── model_features.joblib
├── requirements.txt
└── README.md
```

---

## Machine Learning Approach

### Feature Engineering

* Amount mismatch (`Transaction_Amount - Amount_paid`)
* Vehicle profile (`Vehicle_Type + Vehicle_Dimensions`)
* High-speed detection (> 80 km/h)
* Temporal features (hour, day, month)

### Models Used

* Logistic Regression
* Decision Tree
* Random Forest (final deployed model)
* Support Vector Machine
* K-Nearest Neighbors

---

##  How to Run the Application

### 1️ Clone the repository

```bash
git clone https://github.com/your-username/fastag-fraud-detection.git
cd fastag-fraud-detection
```

### 2️ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️ Train the model (optional)

```bash
python train_model.py
```

### 4️ Run the Streamlit app

```bash
streamlit run app.py
```

---

##  Input Options

### Manual Input

* Vehicle type
* Toll booth ID
* Lane type
* Transaction amount
* Vehicle speed
* Date & time features

### CSV Upload

Required columns:

```
Vehicle_Type
TollBoothID
Lane_Type
Vehicle_Dimensions
Transaction_Amount
Amount_paid
Vehicle_Speed
Hour
DayOfWeek
State_code
```

---

##  Output

* Fraud / Not Fraud prediction
* Downloadable CSV with predictions
* Visual dashboards and analytics

---

##  Future Enhancements

* Real-time streaming integration
* Deep learning–based anomaly detection
* API deployment
* Role-based access control
* Cloud deployment (AWS / GCP)

---

## Author

**Rohith Boggula**
Data Science Aspirant

---

##  License

This project is for **academic and educational purposes only**.

---

