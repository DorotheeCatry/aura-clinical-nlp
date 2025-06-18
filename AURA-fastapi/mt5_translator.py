from transformers import MT5ForConditionalGeneration, MT5Tokenizer

class MT5Translator() :
    def __init__(self):
        # Chargement mT5 (base)
        self.mt5_model_name = "google/mt5-base"
        self.mt5_tokenizer = MT5Tokenizer.from_pretrained(self.mt5_model_name)
        self.mt5_model = MT5ForConditionalGeneration.from_pretrained(self.mt5_model_name)

    def translate_mt5(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Utilise mt5 pour traduire via un prompt de type 'translate {source_lang} to {target_lang}: {text}'
        """
        input_text = f"translate {source_lang} to {target_lang}: {text}"
        inputs = self.mt5_tokenizer(input_text, return_tensors="pt")
        outputs = self.mt5_model.generate(**inputs, max_length=512)
        translated = self.mt5_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated