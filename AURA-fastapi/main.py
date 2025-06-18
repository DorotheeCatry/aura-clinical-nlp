from fastapi import FastAPI
from text_request import TextRequest
from mt5_translator import MT5Translator
from bioBERT_manager import BioBERTManager

# run with : uvicorn main:app --reload 
# then open http://127.0.0.1:8000 
app = FastAPI()

@app.post("/process_text")
async def process_text(request: TextRequest):
    
    # 1. Traduction FR -> EN
    translator = MT5Translator()
    text_en = translator.translate_mt5(request.text, source_lang="fr", target_lang="en")

    # 2. Traitement BioBERT sur l'anglais (exemple: embeddings CLS)
    bioBERT_manager  = BioBERTManager()
    response_en = bioBERT_manager.process(text_en)

    # 3. Traduction EN -> FR
    response_fr = translator.translate_mt5(response_en, source_lang="en", target_lang="fr")

    return {"input_text_fr": request.text,
            "translated_en": text_en,
            "biobert_response_en": response_en,
            "response_fr": response_fr}