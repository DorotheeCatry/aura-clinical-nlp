from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.nn.functional import softmax

# Charge le tokenizer et le modèle depuis Hugging Face Hub
# tokenizer = AutoTokenizer.from_pretrained("waelbensoltana/finetuned-medical-fr")
# model = AutoModelForSequenceClassification.from_pretrained("waelbensoltana/finetuned-medical-fr")
tokenizer = AutoTokenizer.from_pretrained("NicoCasso/finetuned-medical-fr")
model = AutoModelForSequenceClassification.from_pretrained("NicoCasso/finetuned-medical-fr")

# Exemple de texte à classifier
texts =[]

texts.append("Le patient est très diabétique..")
texts.append("Le patient doit surveiller son taux de sucre")
texts.append("J'ai des bourdonnements d'oreilles fréquents")
texts.append("Le patient a une tension élevée")
texts.append("Le diabète de type 2 est difficile à contrôler dans ce cas.")
texts.append("On note une tachycardie persistante suite à une activité physique intense.")
texts.append("La dépression est diagnostiquée depuis plusieurs semaines.")
texts.append("Le diabète de type 2 est difficile à contrôler dans ce cas.")

labels = ['hypertension artérielle', 'troubles psychiques','diabete']

print()
for text in texts :
    # Tokenisation
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Prédiction
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()

    probs_list = softmax(outputs.logits).squeeze().tolist()
    print(f"Texte source : {text} ")
    print(f"Predictions :  {probs_list[0]:.2f}   {probs_list[1]:.2f}   {probs_list[2]:.2f}   => {prediction}") 
    print()
