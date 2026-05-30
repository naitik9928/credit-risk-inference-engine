import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_clean_data(filepath:str)->pd.DataFrame:
    df=pd.read_csv(filepath)

    df=df[df["person_age"]<=100]

    df["person_emp_length"]=df["person_emp_length"].fillna(df["person_emp_length"].median())
    df["loan_int_rate"]=df["loan_int_rate"].fillna(df["loan_int_rate"].median())

    return df

def split_and_save(df:pd.DataFrame,target_col:str):

    x=df.drop(columns=[target_col])
    y=df[target_col]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


    train_dataset=pd.concat([x_train,y_train],axis=1)
    test_dataset=pd.concat([x_test,y_test],axis=1)

    train_dataset.to_csv("train.csv",index=False)
    test_dataset.to_csv("test.csv",index=False)


if __name__=="__main__":
    RAW_DATA_PATH=r"C:\Users\Hp\Desktop\New folder\credit_risk_dataset.csv"
    TARGET_VARIABLE="loan_status"

    cleaned_dataframe=load_and_clean_data(RAW_DATA_PATH)
    split_and_save(cleaned_dataframe,TARGET_VARIABLE)