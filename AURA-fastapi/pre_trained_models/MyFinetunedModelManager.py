from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class MyFinetunedModelManager:
    def __init__(self):
        self.model_name = "FinetunedMedicalModel"
        self.model_path = "results"  # chemin relatif à ton fichier main.py
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

    def generate_fr(self, text: str) -> str:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return f"Classe prédite : {prediction}"