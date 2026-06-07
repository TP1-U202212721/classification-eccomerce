import joblib
import pandas as pd
model = joblib.load('validation/model/estimation_risk_model.pkl')

def predict_risk(data):
    df = pd.DataFrame([data])  
    prediction = model.predict(df)
    return prediction[0]
