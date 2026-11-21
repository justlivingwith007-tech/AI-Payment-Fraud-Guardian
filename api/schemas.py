"""Typed transaction request and prediction response schemas."""
from pydantic import BaseModel, ConfigDict, Field, create_model

fields = {"Time": (float, Field(ge=0, allow_inf_nan=False)),
          **{f"V{i}": (float, Field(allow_inf_nan=False)) for i in range(1, 29)},
          "Amount": (float, Field(ge=0, allow_inf_nan=False))}
Transaction = create_model("Transaction", __config__=ConfigDict(extra="forbid"), **fields)


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_flagged: bool
    threshold: float
    risk_level: str
    model_name: str
