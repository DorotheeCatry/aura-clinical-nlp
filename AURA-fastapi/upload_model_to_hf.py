from huggingface_hub import create_repo, Repository, upload_folder
import os

print(os.getcwd())

# Préalable :
#  créer un login sur hugginface
#  puis dans profil, settings créer un token 

# se connecter à Hugging face avec 
#     huggingface-cli login
#     ça va demander le token

# installer git lfs avec 
# sudo apt update
# sudo apt install git-lfs

# Initialiser Git LFS une fois pour toutes
# git lfs install


# === CONFIGURATION ===
repo_name = "finetuned-medical-fr"
#user = "waelbensoltana"  # ← Remplace par ton nom d'utilisateur HF
user = "NicoCasso"  # ← Remplace par ton nom d'utilisateur HF
local_dir = "notebooks/results"
repo_url = f"https://huggingface.co/{user}/{repo_name}"
full_model_path = os.path.abspath(local_dir)
readme_path = os.path.join(local_dir, "README.md")
repo_id = f"{user}/{repo_name}"

# === 1. Créer le repo sur Hugging Face (s’il n’existe pas) ===
"""create_repo(repo_id=f"{user}/{repo_name}", repo_type="model", private=False, exist_ok=True)"""
create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

# === 2. Créer ou écraser le README.md dans results/ ===
readme_content = f"""---
language: fr
license: apache-2.0
tags:
  - classification
  - medical
  - french
  - fine-tuning
  - torch
  - transformers
  - healthcare
datasets:
  - custom
model-index:
  - name: FinetunedMedicalModel
    results: []
---

# 🧠 FinetunedMedicalModel

Modèle de classification fine-tuné sur des textes médicaux en français. Il a été entraîné avec 🤗 Transformers et PyTorch à partir d’un modèle pré-entraîné.

---

## 📦 Utilisation

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("{user}/{repo_name}")
model = AutoModelForSequenceClassification.from_pretrained("{user}/{repo_name}")

text = "Le patient présente une douleur abdominale persistante."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = outputs.logits.argmax(dim=-1).item()

print("Classe prédite :", prediction)
```
🔬 Détails
Langue : Français 🇫🇷
Tâche : Classification de texte médical
Framework : PyTorch
Base de données : Jeu de données personnalisé (non public)

📄 Licence
Ce modèle est publié sous la licence Apache 2.0 :
✅ libre d’usage, modification, et distribution, même à des fins commerciales.

Pour plus d’infos : https://www.apache.org/licenses/LICENSE-2.0

✨ Auteur
Ce modèle a été entraîné et publié par {user}.
"""
with open(readme_path, "w+", encoding="utf-8") as f:
    f.write(readme_content)

upload_folder(
    repo_id=repo_id, 
    folder_path=local_dir, 
    commit_message="Upload initial avec README")

# force le push ?
# repo = Repository(local_dir=local_dir, clone_from=repo_url)
# repo.push_to_hub()