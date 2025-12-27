import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

# Load Cleaned Data
def load_data(file_path='data/cleaned_data.csv'):
    df = pd.read_csv(file_path)
    # Amount Mismatch
    df['Amount_Mismatch'] = df['Transaction_Amount'] - df['Amount_paid']

    # Vehicle Profile
    df['Vehicle_Profile'] = df['Vehicle_Type'] + '_' + df['Vehicle_Dimensions']

    # High-Speed Flag (optional threshold e.g., 80 km/h)
    df['High_Speed'] = df['Vehicle_Speed'] > 80
    return df



# Label Encoding
def label_encode_data(df):
    label_encoder = {}
    object_columns = ['Vehicle_Type', 'Lane_Type', 'Vehicle_Dimensions', 
                       'TollBoothID', 'State_code', 'Fraud_indicator','Vehicle_Profile']
    for column in object_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoder[column] = le
    return df, label_encoder

# Split Data into Train and Test
def split_data(df):
    X = df.drop(columns=["Fraud_indicator"])
    y = df["Fraud_indicator"]
    print(X)
    print(y)
    return train_test_split(X, y, test_size=0.3, random_state=42)

# Evaluate Model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1 Score: {f1}")

    return accuracy, precision, recall, f1, conf_matrix

# Plot Confusion Matrix
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {title}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Logistic Regression
def logistic_regression(X_train, X_test, y_train, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Logistic Regression Results:")
    return evaluate_model(y_test, y_pred)

# Decision Tree
def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Decision Tree Results:")
    return evaluate_model(y_test, y_pred)

# SVM Classifier
def svm_classifier(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    # print("Support Vector Machine Classifier Results:")
    return evaluate_model(y_test, y_pred)

# Random Forest Classifier
def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("Random Forest Results:")
    return evaluate_model(y_test, y_pred)

# KNN Classifier
def knn_classifier(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # print("KNN Classifier Results:")
    return evaluate_model(y_test, y_pred)

# Model Comparison Plot
def plot_model_comparison(results):
    models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN']
    accuracy_scores = [res[0] for res in results]
    colors = ['blue', 'purple', 'magenta', 'green', 'orange']

    # Create Figure and Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(models, accuracy_scores, color=colors)
    ax.set_xlabel('Machine Learning Models')
    ax.set_ylabel('Accuracy Scores')
    ax.set_title('Comparison of Accuracy Scores of Different Models')

    return fig  # Return the figure instead of plt.show()

