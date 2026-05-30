from fastapi import FastAPI
import pydantic 
import joblib
import pandas as pd 
from pydantic import BaseModel
from logger import logger
app=FastAPI(title="Credict Card Risk Assesment Engine")

model_path="model.pkl"
model_pipeline=joblib.load(model_path)

class ApplicationData(BaseModel):
    person_age: int
    person_income: int
    person_emp_length: float
    loan_amnt: int
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: int
    person_home_ownership: str
    loan_intent: str
    loan_grade: str
    cb_person_default_on_file: str

@app.get("/")
def health_check():
    return {"status":"healthy","model_loaded":True}

@app.post("/predict")
def predict_risk(application:ApplicationData):

    raw_data=application.model_dump()

    input_df=pd.DataFrame([raw_data])

    prediction=model_pipeline.predict(input_df)[0]
    probabilities=model_pipeline.predict_proba(input_df)[0]

    default_probabilities=float(probabilities[1])
    decision="deny credit card" if prediction==1 else "approve credit card"

    logger.info(f"Incoming request -Income:{raw_data["person_income"]},Loan:{raw_data["loan_amnt"]}")
    logger.info(f"Prediction result decision {decision}, Probability {round(default_probabilities,4)}")
    return {
        "risk_assesment_decision":decision,
        "probability_of_default":round(default_probabilities,4)
    }

