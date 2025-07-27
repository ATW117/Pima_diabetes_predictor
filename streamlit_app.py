import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('diabetes_rf_model.pkl')
    return model

# Recreate and fit preprocessor on the fly
@st.cache_resource
def get_fitted_preprocessor():
    # Load the training data to fit the preprocessor
    df = pd.read_csv("diabetes.csv") #Dataset should be in the root directory
    
    # Recreate the preprocessor with current scikit-learn version
    features_to_impute = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    other_features = ["Pregnancies", "DiabetesPedigreeFunction", "Age"]
    
    impute_scale_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([
        ('imp_features', impute_scale_pipeline, features_to_impute),
        ('pass_features', 'passthrough', other_features)
    ])
    
    # Fit the preprocessor with the training data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Fit the preprocessor
    preprocessor.fit(X_train)
    
    return preprocessor

model = load_model()
preprocessor = get_fitted_preprocessor()

# App title
st.title('Diabetes Prediction App')
st.write('This app predicts diabetes risk using a Random Forest model.')

# Sidebar for input
st.sidebar.header('Patient Information')

# Create input fields
pregnancies = st.sidebar.slider('Number of Pregnancies', 0, 17, 1)
glucose = st.sidebar.slider('Glucose Level', 0, 200, 120)
blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
skin_thickness = st.sidebar.slider('Skin Thickness', 0, 99, 20)
insulin = st.sidebar.slider('Insulin Level', 0, 846, 80)
bmi = st.sidebar.slider('BMI', 18.0, 67.1, 25.0)
diabetes_pedigree = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.5)
age = st.sidebar.slider('Age', 21, 81, 30)

# Create a button to make prediction
if st.sidebar.button('Predict Diabetes Risk'):
    # Create input data
    input_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    # Preprocess the input data
    input_processed = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_processed)[0]
    prediction_proba = model.predict_proba(input_processed)[0]
    
    # Display results
    st.header('Prediction Results')
    
    if prediction == 1:
        st.error('⚠️ **High Risk of Diabetes**')
        st.write(f'Probability: {prediction_proba[1]:.2%}')
    else:
        st.success('✅ **Low Risk of Diabetes**')
        st.write(f'Probability: {prediction_proba[0]:.2%}')
    
    # Show probability breakdown
    st.subheader('Risk Breakdown')
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Low Risk Probability", f"{prediction_proba[0]:.2%}")
    
    with col2:
        st.metric("High Risk Probability", f"{prediction_proba[1]:.2%}")
    
    # Show input values
    st.subheader('Input Values')
    st.write(input_data)

# Add some information about the model
st.markdown("---")
st.subheader('About the Model')
st.write("""
This model uses a Random Forest classifier trained on the Pima Indians Diabetes dataset.
The model considers the following factors:
- Number of pregnancies
- Glucose level
- Blood pressure
- Skin thickness
- Insulin level
- BMI (Body Mass Index)
- Diabetes pedigree function
- Age
""")

# Add model performance info
st.subheader('Model Performance')
st.write("""
- **Accuracy**: 76.6%
- **Recall**: 80.0%
- **Precision**: 63.8%
- **F1 Score**: 71.0%
""")
