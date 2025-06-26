from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Charge le tokenizer et le modèle depuis Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("waelbensoltana/finetuned-medical-fr")
model = AutoModelForSequenceClassification.from_pretrained("waelbensoltana/finetuned-medical-fr")

# Exemple de texte à classifier
text = "Le patient est très diabétique.."

# Tokenisation
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Prédiction
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1).item()

print("Classe prédite :", prediction)