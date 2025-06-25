# AURA - Assistant Médical IA

Assistant intelligent pour la surveillance hospitalière avec analyse NLP avancée.

## 🚀 Démarrage rapide

### 1. Configuration de l'environnement

```bash
# Copier le fichier d'exemple
cp .envexample .env

# Modifier les variables selon vos besoins
# FASTAPI_BASE_URL=http://127.0.0.1:8001
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

### 4. Démarrage des services

#### Terminal 1 - FastAPI (Port 8001)
```bash
cd AURA-fastapi
uvicorn main:app --reload --port 8001
```

#### Terminal 2 - Django (Port 8000)
```bash
cd aura_project
python manage.py runserver
```

#### Terminal 3 - Tailwind (développement)
```bash
cd aura_project
python manage.py tailwind start
```

## 🔧 Architecture

### FastAPI (Port 8001)
- **Endpoints disponibles :**
  - `GET /get_available_models` - Liste des modèles IA
  - `POST /process_text` - Traitement de texte par IA

### Django (Port 8000)
- **Interface web complète**
- **API de transcription audio**
- **Gestion des patients et observations**
- **Intégration automatique avec FastAPI**

## 🧠 Pipeline NLP

### Mode FastAPI (Recommandé)
- Classification thématique via vos modèles
- Extraction d'entités médicales
- Génération de résumés automatiques
- Transcription audio locale (Whisper)

### Mode Local (Fallback)
- Simulation intelligente si FastAPI indisponible
- Transcription audio toujours fonctionnelle
- Robustesse maximale

## 📊 Fonctionnalités

- ✅ **Gestion des patients** - CRUD complet
- ✅ **Observations médicales** - Texte et audio
- ✅ **Transcription automatique** - Whisper français
- ✅ **Classification IA** - Spécialités médicales
- ✅ **Extraction d'entités** - Médicaments, symptômes, etc.
- ✅ **Résumés automatiques** - Synthèse intelligente
- ✅ **Dashboard avancé** - Statistiques et monitoring
- ✅ **API REST** - Intégration FastAPI transparente

## 🔗 URLs importantes

- **Django App :** http://127.0.0.1:8000
- **FastAPI Docs :** http://127.0.0.1:8001/docs
- **Admin Django :** http://127.0.0.1:8000/admin

## 🛠️ Développement

### Ajouter un nouveau modèle FastAPI

1. Créer votre manager dans `AURA-fastapi/pre_trained_models/`
2. L'ajouter à `AVAILABLE_MODELS` dans `main.py`
3. Redémarrer FastAPI

### Personnaliser la pipeline NLP

Modifier `aura_project/med_assistant/nlp_pipeline.py` pour adapter les traitements.

## 📝 Licence

MIT License - Voir le fichier LICENSE pour plus de détails.