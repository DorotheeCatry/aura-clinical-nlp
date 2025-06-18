from transformers import AutoModel, AutoTokenizer
import torch

class BioBERTManager() :
    def __init__(self):
        # Chargement BioBERT (anglais)
        self.biobert_model_name = "dmis-lab/biobert-base-cased-v1.1"
        self.biobert_tokenizer = AutoTokenizer.from_pretrained(self.biobert_model_name)
        self.biobert_model = AutoModel.from_pretrained(self.biobert_model_name)

    def process(self, text_en:str) -> str:
        # 2. Traitement BioBERT sur l'anglais (exemple: embeddings CLS)
        inputs = self.biobert_tokenizer(text_en, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.biobert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # vecteur CLS, taille (1, hidden_size)

           # Ici, on fait un traitement fictif : par exemple on prend la norme du vecteur comme "score"
        score = torch.norm(cls_embedding).item()
        response_en = f"The score of the input text is {score:.2f}."

        return response_en