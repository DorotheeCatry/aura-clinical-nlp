from transformers import MT5ForConditionalGeneration, T5Tokenizer, BitsAndBytesConfig
import torch

class MedicalMT5Manager:
    def __init__(self, use_quantization=True, quantization_bits=8):
        self.model_name = "HiTZ/Medical-mT5-large"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Chargement du tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, legacy=False)
        
        # Configuration de la quantification
        if use_quantization and torch.cuda.is_available():
            if quantization_bits == 8:
                # Configuration pour quantification 8-bit
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    llm_int8_skip_modules=None,
                )
            elif quantization_bits == 4:
                # Configuration pour quantification 4-bit (plus agressive)
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                raise ValueError("quantization_bits doit être 4 ou 8")
            
            print(f"Chargement du modèle avec quantification {quantization_bits}-bit...")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map="auto",  # Placement automatique sur GPU/CPU
                low_cpu_mem_usage=True
            )
            print(f"Modèle chargé avec quantification {quantization_bits}-bit sur {self.device}")
            
        else:
            # Chargement standard avec optimisations
            print("Chargement du modèle sans quantification...")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)
            print(f"Modèle chargé sur {self.device}")
        
        # Mode évaluation pour optimiser les performances
        self.model.eval()
        
        # Affichage des informations sur l'utilisation mémoire
        if torch.cuda.is_available():
            print(f"Mémoire GPU utilisée: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Mémoire GPU réservée: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    def generate_fr(self, question, in_max_length=256, out_max_length=256):
        if not question or not question.strip():
            return "pas de question"
        
        input_text = f"explique: {question}"
        if len(input_text) > in_max_length:
            return "question trop longue"
        
        # Tokenisation
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=in_max_length,
            truncation=True,
            padding=True
        )
        
        # Déplacer les inputs sur le bon device si nécessaire
        if hasattr(self.model, 'device'):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
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
    
    def get_memory_usage(self):
        """Affiche l'utilisation actuelle de la mémoire"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            return f"Mémoire GPU - Utilisée: {allocated:.2f} GB, Réservée: {reserved:.2f} GB"
        else:
            return "GPU non disponible"
    
    def clear_cache(self):
        """Nettoie le cache GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("Cache GPU nettoyé")