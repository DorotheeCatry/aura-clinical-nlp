from fastapi import FastAPI
from json_format_requests_and_responses import AuraRequest, AuraResponse, AvailableModelsResponse
from pre_trained_models import GPT2Manager, FacebookMBartManager, MedicalMT5Manager, MistralManager


# run with : uvicorn main:app --reload 
# then open http://127.0.0.1:8000 
app = FastAPI()

def available_models() -> list[str] :
    available_models = []
    available_models.append(GPT2Manager())
    available_models.append(FacebookMBartManager())
    #available_models.append(MedicalMT5Manager()) # inexploitable a cause des balises <input_id>
    #available_models.append(MistralManager() ) # too long to load ??
    available_models : list[str] = available_models
    return available_models

@app.get("/get_available_models")
async def get_available_models() -> AvailableModelsResponse:

    response_list : list[str] = []
    for model in get_available_models() :
        response_list.append(model.model_name)

    json_response = AvailableModelsResponse(model_names= response_list)
    return json_response

@app.post("/process_text")
async def process_text(request: AuraRequest) -> AuraResponse:
    selected = None
    for model in get_available_models() :
        if model.model_name == request.model_name : 
            selected = model
    
    response_text = model.generate_fr(request.question_text)

    return AuraResponse(model_name= model.model_name, response=response_text)


   