from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch

class MedicalMT5Manager:
    def __init__(self):

        self.model_name = "HiTZ/Medical-mT5-large"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        self.model = MT5ForConditionalGeneration.from_pretrained(self.model_name)
        
        print(f"Modèle chargé sur {self.device}")
    
    def generate_fr(self, question, in_max_length=256, out_max_length=256):
        if not question or not question.strip() :
            return "pas de question"

        input_text = f"question: {question}"

        if len(input_text) >  in_max_length:
            return "question trop longue"
        
        # Tokenisation
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=in_max_length,
            truncation=True,
            padding=True
        )
        
        # Génération de la réponse
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=out_max_length,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Décodage de la réponse
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
