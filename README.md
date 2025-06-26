# AURA - Assistant Médical IA

Assistant intelligent pour la surveillance hospitalière avec analyse NLP avancée utilisant des modèles Hugging Face.

## 🚀 Démarrage rapide

### 1. Configuration de l'environnement

```bash
# Copier le fichier d'exemple
cp .envexample .env

# Modifier les variables selon vos besoins
# NLP_USE_HUGGINGFACE=True
```

### 2. Installation des dépendances

```bash
# Installer les dépendances Python
pip install -r requirements.txt

# Installer les dépendances Tailwind
cd aura_project
python manage.py tailwind install
```

### 3. Configuration de la base de données

```bash
cd aura_project
python manage.py migrate
python manage.py createsuperuser
```

### 4. Démarrage du service Django

```bash
cd aura_project
python manage.py runserver
```

### 5. Démarrage de Tailwind (développement)

```bash
cd aura_project
python manage.py tailwind start
```

## 🔧 Architecture

### Django (Port 8000)
- **Interface web complète**
- **API de transcription audio**
- **Gestion des patients et observations**
- **Modèles Hugging Face intégrés**

## 🧠 Pipeline NLP

### Modèles Hugging Face Directs
- **Classification thématique** : `waelbensoltana/finetuned-medical-fr`
- **Extraction d'entités médicales** : `Thibeb/DrBert_generalized`
- **Génération de résumés automatiques** : `plguillou/t5-base-fr-sum-cnndm`
- **Transcription audio** : `openai/whisper-small`

### Optimisations mémoire
- Chargement à la demande des modèles
- Libération automatique après utilisation
- Gestion intelligente du GPU limité
- Fallback en simulation si problème

## 📊 Fonctionnalités

- ✅ **Gestion des patients** - CRUD complet
- ✅ **Observations médicales** - Texte et audio
- ✅ **Transcription automatique** - Whisper français
- ✅ **Classification IA** - Spécialités médicales
- ✅ **Extraction d'entités** - Médicaments, symptômes, etc.
- ✅ **Résumés automatiques** - Synthèse intelligente
- ✅ **Dashboard avancé** - Statistiques et monitoring
- ✅ **Pipeline optimisée** - Gestion mémoire GPU

## 🔗 URLs importantes

- **Django App :** http://127.0.0.1:8000
- **Admin Django :** http://127.0.0.1:8000/admin

## 🛠️ Développement

### Personnaliser la pipeline NLP

Modifier `aura_project/med_assistant/nlp_pipeline.py` pour adapter les traitements ou changer les modèles utilisés.

### Configuration des modèles

Les modèles sont configurés dans la classe `NLPPipeline` :

```python
self.models_config = {
    'classification': 'waelbensoltana/finetuned-medical-fr',
    'entities': 'Thibeb/DrBert_generalized', 
    'summarization': 'plguillou/t5-base-fr-sum-cnndm'
}
```

## 📝 Licence

MIT License - Voir le fichier LICENSE pour plus de détails.