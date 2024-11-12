from pydantic import BaseModel, validator

class PredictRequest(BaseModel):
    query: str
    search_engine: str
    k: int

class PredictResponce(BaseModel):
    status: str
    time_search: float
    search_result: list
    