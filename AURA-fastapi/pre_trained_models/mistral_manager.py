from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class MistralManager():
    def __init__(self):
        self.model_name = "BioMistral/BioMistral-7B"

        # Chargement du tokenizer et du modèle avec options adaptées
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",            # envoie automatiquement sur le GPU si disponible
            torch_dtype="auto",           # FP16 si possible
            trust_remote_code=True
        )
        print(f"Modèle '{self.model_name}' chargé avec succès.")

        # Création du pipeline de génération
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto"
        )

        # Pré-prompt utilisé pour contextualiser
        self.pre_prompt = (
            "Tu es un professionnel de santé. Réponds à la question suivante en français, de façon claire et fiable : "
        )

    def generate_fr(self, prompt_fr: str, max_length=512) -> str:

        # Format [INST] conforme aux modèles Mistral instruct
        full_prompt = f"<s>[INST] {self.pre_prompt}{prompt_fr} [/INST]"

        output = self.generator(
            full_prompt,
            max_new_tokens=max_length,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            num_return_sequences=1
        )

        response = output[0]["generated_text"]

        # Nettoyage : enlever le prompt d'origine
        returned_response =  response.replace(full_prompt, "").strip()
        return returned_response

