# Credit Risk Inference Engine

A FastAPI-based machine learning inference engine for assessing credit card default risk using XGBoost classification. This project provides an automated decision-making system that evaluates loan applications and predicts the probability of default.

## 🎯 Overview

The Credit Risk Inference Engine is a production-ready ML application that:
- **Predicts credit risk** using historical loan data
- **Provides real-time assessments** via REST API endpoints
- **Offers probability scores** for default risk
- **Makes binary decisions** to approve or deny credit applications
- **Runs in containers** for easy deployment

## ✨ Features

- **FastAPI REST API** with automatic OpenAPI documentation
- **XGBoost model** for accurate risk prediction
- **Data preprocessing pipeline** with imputation and scaling
- **Categorical encoding** for mixed data types
- **Structured logging** for monitoring and debugging
- **Docker support** for containerized deployment
- **Type-safe** with Pydantic models

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/naitik9928/credit-risk-inference-engine.git
   cd credit-risk-inference-engine
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if needed)
   ```bash
   # First, prepare your data
   python data_prep.py
   
   # Then train the model
   python train.py
   ```

4. **Run the API server**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

5. **Access the API**
   - API endpoint: `http://localhost:8000`
   - Interactive docs: `http://localhost:8000/docs`
   - Alternative docs: `http://localhost:8000/redoc`

## 📦 Project Structure

```
credit-risk-inference-engine/
├── app.py              # FastAPI application and prediction endpoints
├── train.py            # Model training pipeline
├── data_prep.py        # Data loading, cleaning, and splitting
├── logger.py           # Logging configuration
├── requirements.txt    # Python dependencies
├── DockerFile          # Docker container configuration
├── model.pkl           # Trained XGBoost model (generated after training)
├── train.csv           # Training dataset (generated after data prep)
├── test.csv            # Test dataset (generated after data prep)
└── README.md           # This file
```

## 📚 API Documentation

### Endpoints

#### 1. Health Check
```http
GET /
```
**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Credit Risk Prediction
```http
POST /predict
```

**Request Body:**
```json
{
  "person_age": 35,
  "person_income": 75000,
  "person_emp_length": 10.0,
  "loan_amnt": 15000,
  "loan_int_rate": 8.5,
  "loan_percent_income": 0.2,
  "cb_person_cred_hist_length": 15,
  "person_home_ownership": "RENT",
  "loan_intent": "PERSONAL",
  "loan_grade": "A",
  "cb_person_default_on_file": "N"
}
```

**Response:**
```json
{
  "risk_assesment_decision": "approve credit card",
  "probability_of_default": 0.1234
}
```

**Response Parameters:**
- `risk_assesment_decision`: Binary decision ("approve credit card" or "deny credit card")
- `probability_of_default`: Float value between 0 and 1 representing the likelihood of default

### Input Features

| Feature | Type | Description |
|---------|------|-------------|
| `person_age` | int | Age of the applicant (max: 100) |
| `person_income` | int | Annual income in dollars |
| `person_emp_length` | float | Years of employment |
| `loan_amnt` | int | Loan amount in dollars |
| `loan_int_rate` | float | Interest rate percentage |
| `loan_percent_income` | float | Loan amount as percentage of income |
| `cb_person_cred_hist_length` | int | Years of credit history |
| `person_home_ownership` | str | Home ownership status (RENT, MORTGAGE, OWN) |
| `loan_intent` | str | Purpose of loan (PERSONAL, EDUCATION, MEDICAL, etc.) |
| `loan_grade` | str | Credit grade (A, B, C, D, E, F, G) |
| `cb_person_default_on_file` | str | Previous defaults (Y/N) |

## 🛠️ Data Processing

### Data Preparation (`data_prep.py`)
- **Loads** CSV data from source
- **Cleans** by removing outliers (age > 100)
- **Imputes** missing values:
  - Numeric: median strategy
  - Categorical: most frequent strategy
- **Splits** data into 80% training, 20% testing (stratified)
- **Saves** processed datasets as `train.csv` and `test.csv`

### Model Training (`train.py`)
The training pipeline includes:

1. **Preprocessing:**
   - Numeric transformer: Imputation → Standard scaling
   - Categorical transformer: Imputation → One-hot encoding

2. **Model:** XGBoost classifier with random_state=42

3. **Output:** Pickled model saved as `model.pkl`

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t credit-risk-engine .
```

### Run Container
```bash
docker run -p 8000:8000 credit-risk-engine
```

The API will be available at `http://localhost:8000`

## 📋 Dependencies

- **fastapi**: Modern web framework for building APIs
- **uvicorn**: ASGI web server
- **pydantic**: Data validation using Python type annotations
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning preprocessing and utilities
- **xgboost**: Gradient boosting classifier
- **joblib**: Model serialization and loading

## 📊 Model Performance

The model uses XGBoost with the following characteristics:
- **Algorithm:** Gradient Boosting Classifier
- **Task:** Binary classification (Default: 1, No Default: 0)
- **Output:** Probability scores and binary predictions
- **Training:** Stratified train-test split for balanced evaluation

## 🔍 Logging

The application includes structured logging via `logger.py`:
- **Incoming requests:** Logs key applicant information (income, loan amount)
- **Predictions:** Logs decision and default probability
- **Debug info:** Helps with monitoring and troubleshooting

## 💡 Usage Examples

### Using Python Requests
```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "person_age": 35,
    "person_income": 75000,
    "person_emp_length": 10.0,
    "loan_amnt": 15000,
    "loan_int_rate": 8.5,
    "loan_percent_income": 0.2,
    "cb_person_cred_hist_length": 15,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "loan_grade": "A",
    "cb_person_default_on_file": "N"
}

response = requests.post(url, json=payload)
result = response.json()
print(f"Decision: {result['risk_assesment_decision']}")
print(f"Default Probability: {result['probability_of_default']}")
```

### Using cURL
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "person_age": 35,
    "person_income": 75000,
    "person_emp_length": 10.0,
    "loan_amnt": 15000,
    "loan_int_rate": 8.5,
    "loan_percent_income": 0.2,
    "cb_person_cred_hist_length": 15,
    "person_home_ownership": "RENT",
    "loan_intent": "PERSONAL",
    "loan_grade": "A",
    "cb_person_default_on_file": "N"
  }'
```

## 📝 Workflow

```
Raw Data (CSV)
    ↓
Data Preparation (data_prep.py)
    ├─ Clean & validate
    ├─ Handle missing values
    └─ Split train/test
    ↓
Model Training (train.py)
    ├─ Build preprocessing pipeline
    ├─ Train XGBoost classifier
    └─ Save model (model.pkl)
    ↓
API Server (app.py)
    ├─ Load model
    └─ Serve predictions via REST API
```

## ⚙️ Configuration

Key configuration points in the code:

**Training (`train.py`):**
- Target column: `loan_status`
- Numeric columns: age, income, employment length, loan amount, interest rate, loan percent, credit history
- Categorical columns: home ownership, loan intent, loan grade, default history
- Model: XGBoost with `random_state=42`

**API (`app.py`):**
- Model path: `model.pkl`
- Server port: `8000`
- Host: `0.0.0.0` (in Docker)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Naitik**

GitHub: [@naitik9928](https://github.com/naitik9928)

## 🆘 Troubleshooting

**Model not found error:**
- Ensure `model.pkl` exists in the project root
- Run `train.py` to generate the model

**Port already in use:**
- Change port: `uvicorn app:app --port 8001`

**Import errors:**
- Install dependencies: `pip install -r requirements.txt`

**Docker build issues:**
- Ensure Python 3.10 slim image is available
- Check Dockerfile syntax (note: `workdir` should be `WORKDIR`)

## 📞 Support

For issues or questions, please open an issue on the GitHub repository.

---

**Happy Predicting! 🎯**
