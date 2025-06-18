from pydantic import BaseModel

# Schéma de requête
class TextRequest(BaseModel):
    text: str