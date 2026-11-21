"""HTTP interface to the saved fraud model."""
from fastapi import FastAPI, HTTPException
from api.schemas import Transaction, PredictionResponse
from src.predict import predict_transaction

app = FastAPI(title="AI Payment Fraud Guardian", version="1.0.0",
              description="Educational fraud risk scoring; not for live payment decisions.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        return predict_transaction(transaction.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
