from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Charge le tokenizer et le modèle depuis Hugging Face Hub
# tokenizer = AutoTokenizer.from_pretrained("waelbensoltana/finetuned-medical-fr")
# model = AutoModelForSequenceClassification.from_pretrained("waelbensoltana/finetuned-medical-fr")
tokenizer = AutoTokenizer.from_pretrained("NicoCasso/finetuned-medical-fr")
model = AutoModelForSequenceClassification.from_pretrained("NicoCasso/finetuned-medical-fr")

# Exemple de texte à classifier
texts =[]
texts.append("On note une tachycardie persistante suite à une activité physique intense.")
texts.append("La dépression est diagnostiquée depuis plusieurs semaines.")
texts.append("Le diabète de type 2 est difficile à contrôler dans ce cas.")


print( " ________________")
print("|")

for text in texts :
    # Tokenisation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Prédiction
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()

    print(f"| Texte source : {text}")
    print(f"| Classe prédite : {prediction}")
    print( "|________________")
    print("|")

print()