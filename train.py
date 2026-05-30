import pandas as pd 
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xg

def train_champion_model():
    train_df=pd.read_csv("train.csv")

    x_train=train_df.drop(columns=["loan_status"])
    y_train=train_df["loan_status"]
    num_cols = [
        'person_age', 'person_income', 'person_emp_length', 
        'loan_amnt', 'loan_int_rate', 'loan_percent_income', 
        'cb_person_cred_hist_length'
    ]
    cat_cols = [
        "person_home_ownership", "loan_intent", 
        "loan_grade", "cb_person_default_on_file"
    ]

    num_transformer = Pipeline([
        ("num_imputer", SimpleImputer(strategy="median")),
        ("num_scale", StandardScaler())
    ])
    cat_transformer = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", num_transformer, num_cols),
        ("categorical", cat_transformer, cat_cols)
    ])

    production_pipeline=Pipeline([("process",preprocessor),("clf",xg.XGBClassifier(random_state=42))])
    production_pipeline.fit(x_train,y_train)
    model_filename="model.pkl"
    joblib.dump(production_pipeline,model_filename)

if __name__=="__main__":
    train_champion_model()