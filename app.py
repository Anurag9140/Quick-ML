import streamlit as st
import pandas as pd
import numpy as np
import joblib
from functions import *
import io
import os
from streamlit.watcher.local_sources_watcher import get_module_paths
import sys


def _patched_get_module_paths(module):
            """Skip PyTorch modules to avoid the torch.classes error."""
            if "torch" in module.__name__:
                return []
            return get_module_paths(module)

sys.modules['streamlit.watcher.local_sources_watcher'].get_module_paths = _patched_get_module_paths

st.set_page_config(page_title="QuickML", layout="wide", page_icon="🤖")

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"
if 'preprocessing_params' not in st.session_state:
    st.session_state.preprocessing_params = {}
if 'data_talk_agent' not in st.session_state:
    st.session_state.data_talk_agent = None

def navigation():
    st.sidebar.title("Navigation")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("Auto ML"):
            st.session_state.current_page = "AutoML"
    with col2:
        if st.button("Auto EDA"):
            st.session_state.current_page = "AutoEDA"
    with col3:
        if st.button("Talk with Data"):
            st.session_state.current_page = "TalkWithData"

def home_page():
    st.markdown("<h1 style='text-align: center; color: yellow;'>QuickML</h1>", unsafe_allow_html=True)    
    st.markdown("""
    ## Automated Machine Learning and EDA Platform
    
    **Features:**
    - **Auto ML**: End-to-end automated machine learning pipeline.
    - **Auto EDA**: Comprehensive exploratory data analysis.
    - **Talk with Data**: Chat with your dataset using AI-powered insights.
    
    **Supported Models:**
    - Regression: Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost
    - Classification: Logistic Regression, Random Forest, SVM, XGBoost
    
    **How to Use:**
    1. Use the sidebar buttons to select your task.
    2. Upload your dataset.
    3. Follow the interactive steps.
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/1534/1534959.png", width=200)

def autoeda():
    st.header("Automated Exploratory Data Analysis")
    
    uploaded_file = st.file_uploader("Upload Dataset for EDA", type=["csv", "xlsx"])
    if not uploaded_file:
        return
    
    if 'csv' in uploaded_file.name:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    with st.expander("Dataset Summary"):
        st.write("Shape:", df.shape)
        st.write("Columns:", list(df.columns))
        st.write("Data Types:", df.dtypes)
    
    with st.expander("Missing Values Analysis"):
        fig = plot_missing_values(df)
        st.pyplot(fig)
    
    with st.expander("Distribution Plots"):
        fig = plot_histograms(df)
        st.pyplot(fig)
    
    with st.expander("Correlation Matrix"):
        fig = plot_correlation_matrix(df)
        st.pyplot(fig)
    
    with st.expander("Count Plots"):
        figs = plot_countplots(df)
        for fig in figs:
            st.pyplot(fig)
    
    with st.expander("Box Plots"):
        figs = plot_boxplots(df)
        for fig in figs:
            st.pyplot(fig)
    
    with st.expander("Violin Plots"):
        figs = plot_violinplots(df)
        for fig in figs:
            st.pyplot(fig)
    
    with st.expander("Pair Plot"):
        fig = plot_pairplot(df)
        st.pyplot(fig)
    
    with st.expander("KDE Plots"):
        figs = plot_kde(df)
        for fig in figs:
            st.pyplot(fig)

def automl():
    st.title("Automated Machine Learning Pipeline")
    
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'target_col' not in st.session_state:
        st.session_state.target_col = None
    if 'preprocessed_data' not in st.session_state:
        st.session_state.preprocessed_data = None
    if 'model_info' not in st.session_state:
        st.session_state.model_info = {}
    
    st.header("1. Upload Data")
    uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
    data_type = st.selectbox("Select data type", ["csv", "excel"])
    category = "discrete"
    
    if uploaded_file:
        try:
            if data_type == "csv":
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.data = df
            st.success("Data loaded successfully!")
            st.write("Preview:", df.head())
            
            st.header("2. Handle Missing Values")
            if df.isnull().sum().sum() > 0:
                st.warning("Null values detected!")
                null_action = st.radio("Choose action for null values:", 
                                      ["Remove all", "Replace with mean", "Handle separately"])
                
                if null_action == "Remove all":
                    df = df.dropna()
                elif null_action == "Replace with mean":
                    df = df.fillna(df.mean())
                else:
                    for col in df.columns[df.isnull().any()]:
                        col_action = st.radio(f"Handle {col}", 
                                             ["Remove", "Replace with mean"])
                        if col_action == "Remove":
                            df = df.dropna(subset=[col])
                        else:
                            df[col] = df[col].fillna(df[col].mean())
                st.session_state.data = df
                st.success("Null values handled!")
            
            st.header("3. Handle Outliers")
            numeric_cols = df.select_dtypes(include=np.number).columns
            for col in numeric_cols:
                zscore = (df[col] - df[col].mean()) / df[col].std()
                outliers = df[abs(zscore) > 3]
                if not outliers.empty:
                    st.warning(f"Outliers detected in {col}")
                    st.write(outliers)
                    remove = st.radio(f"Remove outliers in {col}?", ["Yes", "No"])
                    if remove == "Yes":
                        df = df[abs(zscore) <= 3]
            st.session_state.data = df
            
            st.header("4. Select Columns")
            cols_to_drop = st.multiselect("Select columns to drop", df.columns)
            if cols_to_drop:
                df = df.drop(columns=cols_to_drop)
                st.session_state.data = df
                st.success(f"Dropped columns: {', '.join(cols_to_drop)}")
            
            st.header("5. Select Target Column")
            target_col = st.selectbox("Select target column", ["Select a column"] + list(df.columns), index=0)

            if target_col != "Select a column":
                st.session_state.target_col = target_col

            
            st.header("6. Preprocessing")
            df = preprocess_data(df, st.session_state.target_col)
            st.session_state.preprocessed_data = df
            st.success("Preprocessing completed!")
            
            st.header("7. Correlation Analysis")
            corr_matrix = df.corr()
            st.write("Correlation Matrix:", corr_matrix)
                
            low_corr = corr_matrix[target_col][
                (corr_matrix[target_col].abs() < 0.2)].index.tolist()
            if low_corr:
                remove_corr = st.multiselect("Select low-correlation columns to remove", low_corr)
                if remove_corr:
                    df = df.drop(columns=remove_corr)
                    st.session_state.preprocessed_data = df
            
            st.header("8. Model Training")
            X = df.drop(columns=[target_col])
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            problem_type = st.radio("Select problem type", ["Regression", "Classification"])
            if problem_type == "Regression":
                models, top3, scores, top5 = train_regression_models(X_train, X_test, y_train, y_test)
            else:
                models, top3, scores, top5 = train_classification_models(X_train, X_test, y_train, y_test)
                
            tuned_models = hyperparameter_tuning(models, top3, X_train, y_train, problem_type.lower())
            best_model_name = top3[0]
            st.session_state.model = tuned_models[best_model_name]
                
            y_pred = st.session_state.model.predict(X_test)
            st.session_state.model_info = {
                'name': best_model_name,
                'params': tuned_models[best_model_name].get_params(),
                'score': scores[best_model_name],
                'report': classification_report(y_test, y_pred) if problem_type == "Classification" 
                        else f"R2 Score: {r2_score(y_test, y_pred):.4f}"
            }
                
            if st.checkbox("Use ensemble learning?"):
                ensemble = ensemble_learning(models, top3, X_train, y_train, problem_type.lower())
                st.session_state.model = ensemble
                y_pred = ensemble.predict(X_test)
                st.session_state.model_info = {
                    'name': 'Ensemble Model',
                    'params': {m: models[m].get_params() for m in top3},
                    'score': accuracy_score(y_test, y_pred) if problem_type == "Classification"
                            else r2_score(y_test, y_pred),
                    'report': classification_report(y_test, y_pred) if problem_type == "Classification"
                            else f"R2 Score: {r2_score(y_test, y_pred):.4f}"
                }
            
            st.header("9. Results and Model Download")
            if st.session_state.model:
                st.subheader("Model Information")
                st.write(f"**Model Name:** {st.session_state.model_info['name']}")
                st.write("**Best Parameters:**")
                st.json(st.session_state.model_info['params'])
                st.write(f"**Score:** {st.session_state.model_info['score']:.4f}")
                st.write("**Performance Report:**")
                st.text(st.session_state.model_info['report'])
                
                buffer = io.BytesIO()
                model_data = {
                    'model': st.session_state.model,
                    'model_info': st.session_state.model_info,
                    'preprocessing_info': {
                        'target_col': st.session_state.target_col,
                        'category': category,
                        'preprocessing_params': st.session_state.preprocessing_params
                    }
                }
                
                joblib.dump(model_data, buffer)
                buffer.seek(0)
                
                st.download_button(
                    label="⬇️ Download Trained Model",
                    data=buffer,
                    file_name="trained_model.pkl",
                    mime="application/octet-stream",
                    help="Download the trained model with all preprocessing parameters"
                )
            
            st.header("10. Make Predictions")
            if st.checkbox("Make prediction?"):
                if 'model' in st.session_state and 'preprocessed_data' in st.session_state:
                    df = st.session_state.preprocessed_data
                    target_col = st.session_state.target_col
                    
                    st.write("Enter prediction inputs:")
                    input_data = {}
                    
                    for col in df.columns:
                        if col != target_col:
                            if df[col].dtype in [np.int64, np.float64]:
                                input_data[col] = st.number_input(f"Enter value for {col}", value=df[col].mean())
                            else:
                                unique_values = df[col].unique()
                                input_data[col] = st.selectbox(f"Select value for {col}", unique_values)
                    
                    if st.button("Predict"):
                        try:
                            input_df = pd.DataFrame([input_data])
                            
                            input_df = preprocess_data(input_df, target_col)
                            
                            if target_col in input_df.columns:
                                input_df = input_df.drop(columns=[target_col])
                            
                            prediction = st.session_state.model.predict(input_df)
                            
                            st.success(f"Prediction: {prediction[0]}")
                        except Exception as e:
                            st.error(f"Error during prediction: {str(e)}")
                else:
                    st.warning("Please train a model first before making predictions.")
                                
        except Exception as e:
            st.error(f"Error: {str(e)}")


def talk_with_data():
    st.header("Talk with Your Data")
    
    uploaded_file = st.file_uploader("Upload your data file (CSV or Excel)", type=["csv", "xlsx"])
    
    if uploaded_file:
        file_path = f"./uploaded_data{os.path.splitext(uploaded_file.name)[1]}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.session_state.data_talk_agent is None:
            st.session_state.data_talk_agent = create_data_talk_agent(file_path)
        
        st.success("Data uploaded successfully! You can now chat with your data.")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Ask a question about your data"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Thinking..."):
                response = st.session_state.data_talk_agent.run(prompt)
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)


def main():
    navigation()
    
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "AutoEDA":
        autoeda()
    elif st.session_state.current_page == "AutoML":
        automl()
    elif st.session_state.current_page == "TalkWithData":
        talk_with_data()

if __name__ == "__main__":
    main()
