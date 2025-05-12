from pydantic import BaseModel
# --- Vanna Endpoint ---
class NLQueryRequest(BaseModel):
    question: str

class NLQueryResponse(BaseModel):
    question: str
    sql_query: str
    results: list | None
    error: str | None = None