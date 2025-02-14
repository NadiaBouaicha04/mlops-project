import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def prepare_data(data_path='Churn_Modelling.csv'):
    data = pd.read_csv(data_path)

    encoder = LabelEncoder()
    data['State'] = encoder.fit_transform(data['State'])
    data['International plan'] = encoder.fit_transform(data['International plan'])
    data['Voice mail plan'] = encoder.fit_transform(data['Voice mail plan'])
    data['Churn'] = encoder.fit_transform(data['Churn'])

    data = data.drop(['Number vmail messages', 'Total day charge', 'Total eve charge', 
                      'Total night charge', 'Total intl charge'], axis=1)

    X = data.drop(['Churn'], axis=1)
    y = data['Churn']

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    joblib.dump(scaler, 'scaler.joblib')

    return x_train_scaled, x_test_scaled, y_train, y_test

def train_model(X_train, y_train, model_name="Random Forest"):
    models = {
        "Random Forest": RandomForestClassifier(max_depth=20, n_estimators=20, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=9, random_state=42),
        "SVM (RBF Kernel)": SVC(kernel='rbf', C=1000, probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(C=0.01, penalty='l2', solver='liblinear', random_state=42)
    }

    model = models[model_name]
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Rapport de classification:")
    print(classification_report(y_test, y_pred))

    print("Matrice de confusion:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, model_path):
    # Sauvegarde du modèle
    joblib.dump(model, model_path)
    print(f"Modèle sauvegardé dans {model_path}.")

def load_model(model_path):
    return joblib.load(model_path)

