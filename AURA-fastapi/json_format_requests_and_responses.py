from pydantic import BaseModel


class AvailableModelsResponse(BaseModel):
    model_names : list[str]

class AuraRequest(BaseModel):
    requested_model :str
    question_text: str

class AuraResponse(BaseModel):
    model_name : str
    response : str
    
    