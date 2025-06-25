from fastapi import FastAPI
from json_format_requests_and_responses import AuraRequest, AuraResponse, AvailableModelsResponse
from pre_trained_models.MyFinetunedModelManager import MyFinetunedModelManager


# run with : uvicorn main:app --reload --port 8001
# then open http://127.0.0.1:8001 
app = FastAPI()


# On garde la liste dans une variable globale
AVAILABLE_MODELS = [
    #GPT2Manager(),
    #FacebookMBartManager(),
    # MedicalMT5Manager(),  # en commentaire si inutilisable
    # MistralManager(),     # idem
    MyFinetunedModelManager(),  # ton modèle
]

@app.get("/get_available_models")
async def get_available_models() -> AvailableModelsResponse:
    model_names = [model.model_name for model in AVAILABLE_MODELS]
    return AvailableModelsResponse(model_names=model_names)

@app.post("/process_text")
async def process_text(request: AuraRequest) -> AuraResponse:
    selected = next((m for m in AVAILABLE_MODELS if m.model_name == request.requested_model), None)

    if selected is None:
        return AuraResponse(model_name="unknown", response="Modèle non trouvé")

    response_text = selected.generate_fr(request.question_text)
    return AuraResponse(model_name=selected.model_name, response=response_text)