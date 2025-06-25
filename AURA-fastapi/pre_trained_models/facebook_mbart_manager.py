from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class FacebookMBartManager:
    def __init__(self):
        self.model_name = "facebook/mbart-large-50-many-to-many-mmt"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32  # CPU-compatible
        )

        self.src_lang = "fr_XX"
        self.tgt_lang = "fr_XX"

        self.tokenizer.src_lang = self.src_lang

        self.pre_prompt = (
            "Tu es un professionnel de santé. Réponds à la question suivante de façon claire et fiable : "
        )

        print(f"Modèle '{self.model_name}' chargé avec succès.")

    def generate_fr(self, prompt_fr: str, max_new_tokens=128, max_chars=256) -> str:
        full_prompt = self.pre_prompt + prompt_fr

        # Tokenisation avec spécification de la langue source
        inputs = self.tokenizer(full_prompt, return_tensors="pt")

        # Génération avec forçage du token de langue cible
        output_tokens = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[self.tgt_lang]
        )

        response = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Nettoyage : retirer le prompt si répété
        if response.startswith(full_prompt):
            response = response[len(full_prompt):].strip()

        if len(response) > max_chars:
            response = response[:max_chars].rstrip()

        return response
