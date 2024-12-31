import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('XGBoost.pkl')
scaler = joblib.load('scaler.pkl') 

# Define feature names
feature_names = [ "female_age", " E2 ", "AFC ", "Gn_arv", "HCG_E2", "HCG_P", "HCG_neimohoudu ", " Freq"]

## Streamlit user interface
st.title("IVF Predictor")

female_age = st.number_input("Female age:", min_value=18, max_value=100, value=30)
E2= st.number_input("E2 (pg/ml):", min_value=0.00, max_value=500.00, value=150.00)
AFC= st.number_input("AFC:", min_value=0, max_value=100, value=10)
Gn_arv=st.number_input("Gn average dosage (IU/ml):",min_value=0, max_value=500, value=100)
HCG_E2= st.number_input("E2 on HCG day (pg/ml):", min_value=0.00, max_value=10000.00, value=100.00)
HCG_P= st.number_input("P on HCG day (mmol/L):", min_value=0.00, max_value=10.00, value=1.00)
HCG_neimohoudu= st.number_input("Endometrial thickness on HCG day (mm):", min_value=0.00, max_value=50.00, value=10.00)
Freq= st.number_input("Oocyte extracts Frequency:", min_value=1, max_value=10, value=1)

# Process inputs and make predictions
feature_values = [ female_age, E2 , AFC ,  Gn_arv , HCG_E2, HCG_P, HCG_neimohoudu ,  Freq]
features = np.array([feature_values])

if st.button("Predict"):    
    # 标准化特征
    standardized_features = scaler.transform(features)

    # Predict class and probabilities    
    predicted_class = model.predict(standardized_features)[0]   
    predicted_proba = model.predict_proba(standardized_features)[0]

    # Display prediction results    
    st.write(f"**Predicted Class:** {predicted_class} (1: Disease, 0: No Disease)")   
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # Generate advice based on prediction results  
    probability = predicted_proba[predicted_class] * 100
    
    if predicted_class == 1:       
        advice = (
            f"According to our model, your probability of having a successful pregnancy is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "            
            "I recommend regular check-ups to monitor your health, "            
            "and to seek medical advice promptly if you experience any symptoms." 
        )

              
    else:        
        advice = (
            f"According to our model, your probability of not having a successful pregnancy is {probability:.1f}%. "            
            "While this is just an estimate,it suggests that your probability of successful pregnancy are low."              
            "I recommend that you consult a specialist as soon as possible for further evaluation and "            
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )   
        
    st.write(advice)

# Calculate SHAP values and display force plot 
    st.subheader("SHAP Force Plot Explanation") 
    explainer = shap.TreeExplainer(model) 
   
    shap_values = explainer.shap_values(pd.DataFrame(standardized_features, columns=feature_names))
   # 将标准化前的原始数据存储在变量中
    original_feature_values = pd.DataFrame(features, columns=feature_names)

    shap.force_plot(explainer.expected_value, shap_values[0], original_feature_values, matplotlib=True)   
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png", caption='SHAP Force Plot Explanation')
