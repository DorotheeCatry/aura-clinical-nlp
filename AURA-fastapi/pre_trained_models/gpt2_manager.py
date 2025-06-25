from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class GPT2Manager():
    def __init__(self):
        self.model_name = "dbddv01/gpt2-french-small"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Forcer le chargement sur CPU
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32  # float32 car CPU ne gère pas bien le float16
        )

        print(f"Modèle '{self.model_name}' chargé avec succès sur CPU.")

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=-1,  # pipeline CPU
            pad_token_id=self.tokenizer.eos_token_id
        )

        self.pre_prompt = (
            "Tu es un professionnel de santé. Réponds à la question suivante en français, de façon claire et fiable : "
        )

    def generate_fr(self, prompt_fr: str, max_new_tokens=128, max_chars=256) -> str:
        full_prompt = self.pre_prompt + prompt_fr

        output = self.generator(
            full_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = output[0]["generated_text"]

        # Nettoyage : retirer le prompt de la réponse
        if response.startswith(full_prompt):
            response = response[len(full_prompt):].strip()

        # Tronquer à max_chars caractères
        if len(response) > max_chars:
            response = response[:max_chars].rstrip()

        return response
