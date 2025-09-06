import pandas as pd
import numpy as np
import joblib
import re
import string
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, roc_auc_score, roc_curve
)
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier,
    GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, VotingRegressor
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns


# Suppress warnings
warnings.filterwarnings("ignore")

# Helper function to log progress
def log_progress(message):
    print(f"\n🔹 [Progress] {message}")

def handle_nulls(data):
    if data.isnull().any().any():
        null_temp = {column: data[column].isnull().sum() for column in data.columns if data[column].isnull().any()}
        print("Following columns have null values:")
        for key, value in null_temp.items():
            print(f"{key} has {value} null values")

        action = input("What do you want to do with null values? (RemoveALL, ReplaceALLwithMean, SeperateONaLL): ").strip().lower()
        if action == "removeall":
            data = data.dropna()
        elif action == "replaceallwithmean":
            data = data.fillna(data.mean())
        elif action == "seperateonall":
            for column in null_temp:
                action_col = input(f"What do you want to do with {column} column? (RemoveALL, ReplaceALLwithMean): ").strip().lower()
                if action_col == "removeall":
                    data = data.dropna(subset=[column])
                elif action_col == "replaceallwithmean":
                    data[column] = data[column].fillna(data[column].mean())
        else:
            print("Invalid action. Null values will remain unchanged.")
    return data

def detect_outliers(data):
    for column in data.select_dtypes(include=['number']).columns:
        threshold = 3
        zscore = (data[column] - data[column].mean()) / data[column].std()
        outliers = data[abs(zscore) > threshold]
        if not outliers.empty:
            print(f"Outliers detected in {column}:")
            print(outliers)
            action = input("Do you want to remove these outliers? (YES or NO): ").strip().lower()
            if action == "yes":
                data = data[abs(zscore) <= threshold]
    return data

def preprocess_data(data, target_col):
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    # Label Encoding for categorical columns
    le = LabelEncoder()
    for col in categorical_cols:
        if col != target_col and len(data[col].unique()) <= 3:
            data[col] = le.fit_transform(data[col])
        elif col != target_col:
            data = pd.get_dummies(data, columns=[col])

    # Standard Scaling for numerical columns
    ss = StandardScaler()
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    data[numerical_cols] = ss.fit_transform(data[numerical_cols])

    # Label Encoding for target column if categorical
    if target_col in categorical_cols:
        data[target_col] = le.fit_transform(data[target_col])

    return data

def preprocess_text_data(data, target_col):

    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
    
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()
    
    def preprocess_text_temp(text):
        cleaned = clean_text(text)
        words = word_tokenize(cleaned)
        filtered = [word for word in words if word not in stop_words]
        lemmatized = [lemmatizer.lemmatize(word) for word in filtered]
        return " ".join(lemmatized)
    
    # for col in data.select_dtypes(include=['object']).columns:
    #     if col != target_col:
    #         data[col] = data[col].astype(str).apply(preprocess_text_temp)
    
    # Encode target column
    if data[target_col].dtype == 'object':
        le = LabelEncoder()
        data[target_col] = le.fit_transform(data[target_col])

    # Load BERT components
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    # Generate BERT embeddings for text
    text_column = data.columns[data.columns != target_col][0]  # Get first non-target text column
    
    def get_embeddings(texts):
        inputs = tokenizer(texts.tolist(), return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).numpy()
    
    # Replace text with BERT embeddings
    embeddings = get_embeddings(data[text_column])
    data = pd.DataFrame(embeddings)
    data[target_col] = data[target_col]  # Preserve target column
    
    # Store BERT components for later predictions
    joblib.dump({'tokenizer': tokenizer, 'bert_model': bert_model}, 'bert_components.joblib')
    
    return data    
    
def generate_correlation_matrix(data, target_col):
    # Check if the target column is numeric
    if data[target_col].dtype.kind in 'iuf':  # 'iuf' stands for integer, unsigned integer, float
        corr_matrix = data.corr()
        print(f"\nCorrelation between features and {target_col}:")
        print(corr_matrix[target_col].drop(target_col))

        # Identify low-correlation columns
        filtered_correlations = corr_matrix[target_col][(corr_matrix[target_col] > -0.2) & (corr_matrix[target_col] < 0.2)]
        print("These are the columns with low correlation:")
        print(filtered_correlations)

        # Ask user if they want to remove low-correlation columns
        if not filtered_correlations.empty:
            action = input("Do you want to remove these columns? (YES or NO): ").strip().lower()
            if action == "yes":
                cols_to_delete = []
                for col in filtered_correlations.index:
                    delete = input(f"Do you want to delete '{col}'? (YES/NO): ").strip().lower()
                    if delete == "yes":
                        cols_to_delete.append(col)
                data = data.drop(columns=cols_to_delete)
                print(f"Deleted columns: {cols_to_delete}")
        else:
            print("No low-correlation columns to remove.")
    else:
        print("Target column is not numeric. Skipping correlation analysis.")

    return data

def train_regression_models(X_train, X_test, y_train, y_test):
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(),
        "DecisionTree": DecisionTreeRegressor(),
        "RandomForest": RandomForestRegressor(),
        "GradientBoosting": GradientBoostingRegressor(),
        "SVR": SVR(),
        "XGBoost": XGBRegressor()
    }

    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        model_scores[name] = r2

    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    top_5_models = sorted_models[:5]
    top_3_models = [model[0] for model in sorted_models[:3]]

    print("\n🔹 **Top 5 Models and their R² Scores:**")
    for name, score in top_5_models:
        print(f"{name}: {score:.4f}")

    return models, top_3_models, model_scores, top_5_models

def train_classification_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "AdaBoost": AdaBoostClassifier(),
        "SVC": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB(),
        "XGBoost": XGBClassifier(eval_metric='mlogloss')
    }

    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_scores[name] = acc

    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    top_5_models = sorted_models[:5]
    top_3_models = [model[0] for model in sorted_models[:3]]

    print("\n🔹 **Top 5 Models and their Accuracy:**")
    for name, score in top_5_models:
        print(f"{name}: {score:.4f}")

    return models, top_3_models, model_scores, top_5_models

def train_text_ml_models(X_train, X_test, y_train, y_test):
    models = {
        "LogisticRegression": LogisticRegression(),
        "RandomForest": RandomForestClassifier(),
        "SVC": SVC(),
        "XGBoost": XGBClassifier()
    }
    
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        model_scores[name] = accuracy_score(y_test, y_pred)

    
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    top_5_models = sorted_models[:5]
    top_3_models = [model[0] for model in sorted_models[:3]]
    
    print("\n🔹 **Top 5 Models and their Accuracy:**")
    for name, score in top_5_models:
        print(f"{name}: {score:.4f}")
    
    return models, top_3_models, model_scores, top_5_models

def get_bert_embeddings(texts, tokenizer, bert_model, max_length=128):
    inputs = tokenizer(texts.tolist(), return_tensors='pt', truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

def train_text_dl_models(X_train, y_train):
    # Simple classifier on BERT embeddings
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_type, vectorizer=None):
    if model_type == 'classification' and vectorizer is not None:
        X_test = vectorizer.transform(X_test.iloc[:, 0])
    y_pred = model.predict(X_test)
    if model_type == 'classification':
        print("\n🔹 **Classification Report:**")
        print(classification_report(y_test, y_pred))
        
        print("\n🔹 **Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)
            
            # Handle binary and multi-class classification
            if y_proba.shape[1] == 2:  # Binary classification
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                print("\n🔹 **ROC-AUC Score (Binary):**")
                print(f"ROC-AUC Score: {roc_auc:.4f}")
                
                fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve (Binary)')
                plt.legend()
                plt.show()
            else:  # Multi-class classification
                roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
                print("\n🔹 **ROC-AUC Score (Multi-Class):**")
                print(f"ROC-AUC Score (Weighted): {roc_auc:.4f}")
    else:
        print("\n🔹 **Regression Metrics:**")
        print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
        print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.4f}")
        print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred):.4f}")

def detect_overfitting(model, X_train, y_train, X_test, y_test, threshold=0.1):
    """
    Detect overfitting by comparing training and testing accuracy.
    If the difference is greater than the threshold, overfitting is detected.
    """
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"\n🔹 **Training Score:** {train_score:.4f}")
    print(f"🔹 **Testing Score:** {test_score:.4f}")
    
    if abs(train_score - test_score) > threshold:
        print("🔹 **Overfitting Detected!**")
        return True
    else:
        print("🔹 **No Overfitting Detected.**")
        return False

def retrain_model(model, X_train, y_train, X_test, y_test):
    """
    Retrain the model with regularization or other techniques to reduce overfitting.
    """
    if isinstance(model, LogisticRegression):
        print("🔹 **Retraining LogisticRegression with increased regularization (C=0.1)**")
        model = LogisticRegression(C=0.1)
    elif isinstance(model, DecisionTreeClassifier) or isinstance(model, DecisionTreeRegressor):
        print("🔹 **Retraining DecisionTree with reduced max_depth (max_depth=5)**")
        model = DecisionTreeClassifier(max_depth=5) if isinstance(model, DecisionTreeClassifier) else DecisionTreeRegressor(max_depth=5)
    elif isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
        print("🔹 **Retraining RandomForest with reduced max_depth (max_depth=10)**")
        model = RandomForestClassifier(max_depth=10) if isinstance(model, RandomForestClassifier) else RandomForestRegressor(max_depth=10)
    elif isinstance(model, GradientBoostingClassifier) or isinstance(model, GradientBoostingRegressor):
        print("🔹 **Retraining GradientBoosting with reduced learning_rate (learning_rate=0.1)**")
        model = GradientBoostingClassifier(learning_rate=0.1) if isinstance(model, GradientBoostingClassifier) else GradientBoostingRegressor(learning_rate=0.1)
    elif isinstance(model, XGBClassifier) or isinstance(model, XGBRegressor):
        print("🔹 **Retraining XGBoost with reduced learning_rate (learning_rate=0.1)**")
        model = XGBClassifier(learning_rate=0.1) if isinstance(model, XGBClassifier) else XGBRegressor(learning_rate=0.1)
    else:
        print("🔹 **No specific retraining strategy for this model. Using the same model.**")
    
    model.fit(X_train, y_train)
    return model

def hyperparameter_tuning(models, top_3_models, X_train, y_train, model_type):
    """
    Perform hyperparameter tuning on the top 3 models.
    """
    tuned_models = {}
    for name in top_3_models:
        model = models[name]
        if name == "LogisticRegression":
            param_grid = {'C': [0.1, 1, 10], 'penalty': ['l2']}  # Only 'l2' penalty for LogisticRegression
        elif name == "RandomForest":
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
        elif name == "SVC":
            param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        elif name == "XGBoost":
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9]}
        elif name == "LinearRegression":
            param_grid = {'fit_intercept': [True, False]}
        elif name == "Ridge":
            param_grid = {'alpha': [0.1, 1, 10]}
        elif name == "Lasso":
            param_grid = {'alpha': [0.1, 1, 10]}
        elif name == "ElasticNet":
            param_grid = {'alpha': [0.1, 1, 10], 'l1_ratio': [0.5, 0.7, 0.9]}
        elif name == "DecisionTree":
            param_grid = {'max_depth': [None, 10, 20]}
        elif name == "GradientBoosting":
            param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [3, 6, 9]}
        elif name == "AdaBoost":
            param_grid = {'n_estimators': [50, 100, 200]}
        elif name == "KNN":
            param_grid = {'n_neighbors': [3, 5, 7]}
        elif name == "NaiveBayes":
            param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7]}
        else:
            param_grid = {}
        
        if param_grid:
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy' if model_type == 'classification' else 'r2')
            grid_search.fit(X_train, y_train)
            tuned_models[name] = grid_search.best_estimator_
            print(f"\n🔹 **Best parameters for {name}:** {grid_search.best_params_}")
        else:
            tuned_models[name] = model
    
    return tuned_models

def ensemble_learning(models, top_3_models, X_train, y_train, model_type):
    estimators = [(name, models[name]) for name in top_3_models]
    
    if model_type == "regression":
        ensemble = VotingRegressor(estimators=estimators)
    else:
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
    
    ensemble.fit(X_train, y_train)
    return ensemble

def autoforge(data, data_type='csv', category='discrete'):
    if data_type == 'csv':
        data = pd.read_csv(data)
    elif data_type == 'excel':
        data = pd.read_excel(data)
    log_progress("Data loaded successfully")

    # Handle null values
    log_progress("Handling null values")
    data = handle_nulls(data)

    # Detect and handle outliers
    log_progress("Detecting outliers")
    data = detect_outliers(data)

    # Column deletion
    log_progress("Column deletion phase")
    print("Columns in your data:", data.columns.tolist())
    action = input("Do you wish to delete any columns? (YES/NO): ").strip().lower()
    if action == "yes":
        cols_to_delete = input("Enter column names to delete (comma-separated): ").strip().split(',')
        data = data.drop(columns=[c.strip() for c in cols_to_delete])
        log_progress(f"Columns deleted: {cols_to_delete}")

    # Target column selection
    log_progress("Target column selection")
    while True:
        target_col = input("Which is your target column? ").strip().strip("'\"")
        if target_col in data.columns:
            break
        print(f"Error: '{target_col}' not found. Valid columns: {data.columns.tolist()}")

    # Preprocessing
    log_progress("Preprocessing data")
    if category == 'text':
        data = preprocess_text_data(data, target_col)
    else:
        data = preprocess_data(data, target_col)

    # Correlation analysis (skip for text data)
    if category != 'text':
        log_progress("Generating correlation matrix")
        data = generate_correlation_matrix(data, target_col)

    # Train-test split
    log_progress("Splitting data into train/test sets")
    X = data.drop(columns=[target_col])
    y = data[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Model type selection
    if category == 'text':
        log_progress("Model type selection")
        while True:
            model_type = input("Do you want to use ML models or DL models? (ML/DL): ").strip().lower()
            if model_type in ['ml', 'dl']:
                break
            print("Invalid input. Please type 'ML' or 'DL'")

        if model_type == 'ml':
            log_progress("Training text ML models")
            models, top_3_models, model_scores, top_5_models, vectorizer = train_text_ml_models(X_train, X_test, y_train, y_test)
                
            # Evaluate the best model
            best_model_name = top_3_models[0]
            best_model = models[best_model_name]
            log_progress(f"Evaluating the best model: {best_model_name}")
            evaluate_model(best_model, X_test, y_test, model_type='classification', vectorizer=vectorizer)
                
            # Save the best model
            final_model_info = {
                "model": best_model,
                "top_5_models": top_5_models,
                "vectorizer": vectorizer
            }
            joblib.dump(final_model_info, "best_text_ml_model.joblib")
            print("\nBest model saved as 'best_text_ml_model.joblib'")
        else:
            log_progress("Training text DL models")
            final_model, tokenizer, bert_model = train_text_dl_models(X_train, X_test, y_train, y_test)
            final_model_info = {
                "model": final_model,
                "tokenizer": tokenizer,
                "bert_model": bert_model
            }
            joblib.dump(final_model_info, "best_text_dl_model.joblib")
            print("\nBest model saved as 'best_text_dl_model.joblib'")
    else:
        log_progress("Model type selection")
        while True:
            model_type = input("Is this a Regression or Classification problem? (Regression/Classification): ").strip().lower()
            if model_type in ['regression', 'classification']:
                break
            print("Invalid input. Please type 'Regression' or 'Classification'")

        if model_type == 'regression':
            log_progress("Training regression models")
            models, top_3_models, model_scores, top_5_models = train_regression_models(X_train, X_test, y_train, y_test)
            
            # Hyperparameter tuning for discrete data
            log_progress("Hyperparameter tuning on top 3 models")
            tuned_models = hyperparameter_tuning(models, top_3_models, X_train, y_train, model_type='regression')
            
            # Evaluate the best model
            best_model_name = top_3_models[0]
            best_model = tuned_models[best_model_name]
            log_progress(f"Evaluating the best model: {best_model_name}")
            evaluate_model(best_model, X_test, y_test, model_type='regression')
            
            # Overfitting detection and retraining
            if detect_overfitting(best_model, X_train, y_train, X_test, y_test):
                log_progress("Retraining the best model to reduce overfitting")
                best_model = retrain_model(best_model, X_train, y_train, X_test, y_test)
                evaluate_model(best_model, X_test, y_test, model_type='regression')
            
            # Save the best model
            final_model_info = {
                "model": best_model,
                "top_5_models": top_5_models,
                "model_scores": model_scores
            }
            joblib.dump(final_model_info, "best_regression_model.joblib")
            print("\nBest model saved as 'best_regression_model.joblib'")
        else:
            log_progress("Training classification models")
            models, top_3_models, model_scores, top_5_models = train_classification_models(X_train, X_test, y_train, y_test)
            
            # Hyperparameter tuning for discrete data
            log_progress("Hyperparameter tuning on top 3 models")
            tuned_models = hyperparameter_tuning(models, top_3_models, X_train, y_train, model_type='classification')
            
            # Evaluate the best model
            best_model_name = top_3_models[0]
            best_model = tuned_models[best_model_name]
            log_progress(f"Evaluating the best model: {best_model_name}")
            evaluate_model(best_model, X_test, y_test, model_type='classification')
            
            # Overfitting detection and retraining
            if detect_overfitting(best_model, X_train, y_train, X_test, y_test):
                log_progress("Retraining the best model to reduce overfitting")
                best_model = retrain_model(best_model, X_train, y_train, X_test, y_test)
                evaluate_model(best_model, X_test, y_test, model_type='classification')
            
            # Save the best model
            final_model_info = {
                "model": best_model,
                "top_5_models": top_5_models,
                "model_scores": model_scores
            }
            joblib.dump(final_model_info, "best_classification_model.joblib")
            print("\nBest model saved as 'best_classification_model.joblib'")

    # Ask user if they want to use ensemble learning (skip for text data)
    if category != 'text':
        log_progress("Ensemble learning")
        action = input("Do you want to use ensemble learning with the top 3 models? (YES/NO): ").strip().lower()
        if action == "yes":
            ensemble = ensemble_learning(models, top_3_models, X_train, y_train, model_type=model_type)
            y_pred = ensemble.predict(X_test)
            if model_type == 'regression':
                r2 = r2_score(y_test, y_pred)
                print(f"\n🔹 **Ensemble Model R² Score:** {r2:.4f}")
            else:
                acc = accuracy_score(y_test, y_pred)
                print(f"\n🔹 **Ensemble Model Accuracy:** {acc:.4f}")

# Example usage
# autoforge("/content/sentiment_analysis.csv", data_type='csv',category='text')



def dataset_summary(df):
    summary = {
        "Shape": df.shape,
        "Columns": list(df.columns),
        "Missing Values": df.isnull().sum().to_dict(),
        "Data Types": df.dtypes.to_dict(),
        "Summary Statistics": df.describe().to_dict()
    }
    return summary

import missingno as msno
import matplotlib.pyplot as plt

def plot_missing_values(df):
    fig, ax = plt.subplots()
    msno.heatmap(df, ax=ax)
    return fig
import numpy as np

def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]
import seaborn as sns
import matplotlib.pyplot as plt

def plot_histograms(df):
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns
    fig, ax = plt.subplots(figsize=(12, 6))
    numeric_df.hist(ax=ax, bins=30)
    return fig

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    return fig

def plot_countplots(df):
    categorical_cols = df.select_dtypes(include=['object']).columns
    figs = []
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x=col, ax=ax)
        ax.set_title(f"Count Plot of {col}")
        figs.append(fig)
    return figs

def plot_boxplots(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    figs = []
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=df, x=col, ax=ax)
        ax.set_title(f"Box Plot of {col}")
        figs.append(fig)
    return figs

def plot_violinplots(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    figs = []
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.violinplot(data=df, x=col, ax=ax)
        ax.set_title(f"Violin Plot of {col}")
        figs.append(fig)
    return figs

def plot_pairplot(df):
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns
    fig = sns.pairplot(numeric_df)
    return fig

def plot_kde(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    figs = []
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.kdeplot(df[col], ax=ax, fill=True)
        ax.set_title(f"KDE Plot of {col}")
        figs.append(fig)
    return figs





from langchain.document_loaders import CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import pandas as pd
import os

def create_data_talk_agent(file_path):
    file_ext = os.path.splitext(file_path)[1]
    
    if file_ext == '.csv':
        loader = CSVLoader(file_path=file_path)
        documents = loader.load()
    elif file_ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
        documents = [{"content": row.to_json()} for _, row in df.iterrows()]
    else:
        raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    if file_ext == '.csv':
        docs = text_splitter.create_documents([doc.page_content for doc in documents])
    else:
        docs = text_splitter.create_documents([doc['content'] for doc in documents])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_documents(docs, embeddings)

    llm=ChatGroq(api_key="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",model_name="gemma2-9b-it")  
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    data_talk_tool = Tool(
        name="data_talk_tool",
        func=qa_chain.run,
        description="Use this tool to ask questions about the uploaded CSV/Excel data."
    )

    prompt_template = PromptTemplate(
        input_variables=["input"],
        template=(
            "You are a data analyzer expert. Your task is to analyze the given data and answer questions "
            "using mathematical reasoning, logical analysis, and any necessary computations. "
            "Answer as an expert, using all available information from the data and the LLM.\n\n"
            "Question: {input}\n"
            "Answer as a data expert:"
        )
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=[data_talk_tool],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        agent_executor_kwargs={"prompt": prompt_template},
        handle_parsing_errors=True
    )

    return agent

