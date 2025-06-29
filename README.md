# AURA - Assistant Médical IA 🏥

Assistant intelligent pour la surveillance hospitalière avec analyse NLP avancée utilisant des modèles Hugging Face.

## 🚀 Démarrage rapide

### 1. Prérequis
- Python 3.10+
- GPU NVIDIA recommandé (optionnel, fonctionne aussi en CPU)
- 8GB RAM minimum, 16GB recommandé

### 2. Installation

```bash
# Cloner le projet
git clone https://github.com/DorotheeCatry/aura.git
cd aura

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Installer les dépendances
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copier le fichier d'exemple
cp .envexample .env

# Modifier les variables selon vos besoins
# NLP_USE_HUGGINGFACE=True pour activer les modèles IA
# NLP_FALLBACK_TO_LOCAL=True pour le fallback en cas d'erreur
```

### 4. Configuration de la base de données

```bash
cd aura_project
python manage.py migrate
python manage.py create_initial_users  # Créer les utilisateurs de test
```

### 5. Installation de Tailwind CSS

```bash
# Installer les dépendances Tailwind
python manage.py tailwind install
```

### 6. Démarrage des services

```bash
# Terminal 1 - Django (port 8000)
python manage.py runserver

# Terminal 2 - Tailwind (développement uniquement)
python manage.py tailwind start
```

## 🔧 Architecture technique

### Backend Django
- **Interface web complète** avec authentification
- **API de transcription audio** en temps réel
- **Gestion des patients et observations** avec CRUD complet
- **Pipeline NLP intégrée** avec modèles Hugging Face

### Pipeline NLP optimisée
- **Chargement à la demande** des modèles pour économiser la mémoire
- **Gestion intelligente du GPU** avec libération automatique
- **Fallback en simulation** si problème de connexion
- **Cache GPU optimisé** pour les environnements limités

## 🧠 Modèles IA utilisés

| Fonction | Modèle | Description |
|----------|--------|-------------|
| **Classification** | `NicoCasso/finetuned-medical-fr` | Classification automatique par spécialité médicale |
| **Entités médicales** | `Thibeb/DrBert_generalized` | Extraction de médicaments, symptômes, anatomie |
| **Résumés** | `plguillou/t5-base-fr-sum-cnndm` | Génération automatique de résumés |
| **Transcription** | `openai/whisper-small` | Transcription audio vers texte en français |

## 📊 Fonctionnalités principales

### ✅ Gestion des patients
- CRUD complet avec recherche avancée
- Filtrage par spécialité médicale
- Historique des consultations
- Statistiques par patient

### ✅ Observations médicales
- Saisie texte et enregistrement audio
- Transcription automatique Whisper
- Classification IA par spécialité
- Extraction d'entités médicales
- Génération de résumés automatiques

### ✅ Dashboard intelligent
- Métriques en temps réel
- Graphiques interactifs
- Filtres par spécialité
- Activité hebdomadaire

### ✅ Statistiques avancées
- Analyse des médicaments prescrits
- Pathologies les plus fréquentes
- Gestes et procédures médicales
- Regroupement intelligent des entités similaires

## 🔗 Accès à l'application

- **Application principale :** http://127.0.0.1:8000
- **Interface d'administration :** http://127.0.0.1:8000/admin

### Comptes de test créés automatiquement

| Utilisateur | Mot de passe | Rôle |
|-------------|--------------|------|
| `dr.martin` | `aura2024!` | Médecin généraliste (Admin) |
| `inf.dubois` | `aura2024!` | Infirmière |
| `psy.bernard` | `aura2024!` | Psychologue |

## 🛠️ Développement

### Structure du projet
```
aura_project/
├── med_assistant/          # App principale
│   ├── models.py          # Modèles Patient, Observation, UserProfile
│   ├── views.py           # Vues Django avec logique métier
│   ├── nlp_pipeline.py    # Pipeline NLP avec modèles Hugging Face
│   ├── forms.py           # Formulaires Django
│   └── templates/         # Templates HTML avec Tailwind CSS
├── static/                # Fichiers statiques
└── manage.py             # Point d'entrée Django
```

### Personnaliser la pipeline NLP

Modifier `aura_project/med_assistant/nlp_pipeline.py` :

```python
# Configuration des modèles
self.models_config = {
    'classification': 'NicoCasso/finetuned-medical-fr',
    'entities': 'Thibeb/DrBert_generalized', 
    'summarization': 'plguillou/t5-base-fr-sum-cnndm'
}
```

### Ajouter de nouvelles spécialités

Modifier `aura_project/med_assistant/models.py` :

```python
THEME_CHOICES = [
    ('cardiovasculaire', 'Cardiovasculaire'),
    ('psy', 'Psychique/Neuropsychiatrique'),
    ('diabete', 'Métabolique/Diabète'),
    # Ajouter ici vos nouvelles spécialités
]
```

### Tests et débogage

```bash
# Lancer les tests
python manage.py test

# Mode debug avec logs détaillés
DEBUG=True python manage.py runserver

# Vérifier le statut de la pipeline NLP
# Accessible via l'interface web ou l'API /aura/api/nlp/status/
```

## 🔒 Sécurité et conformité

- **Authentification sécurisée** avec gestion des rôles
- **Traçabilité complète** des actions utilisateurs
- **Données médicales chiffrées** en base
- **Conformité RGPD** avec gestion des consentements
- **Logs d'audit** pour toutes les opérations sensibles

## 🚀 Déploiement en production

### Variables d'environnement importantes
```bash
SECRET_KEY="votre-clé-secrète-django"
DEBUG=False
NLP_USE_HUGGINGFACE=True
NLP_FALLBACK_TO_LOCAL=True
```

### Optimisations recommandées
- **GPU NVIDIA** pour accélérer les modèles IA
- **Redis** pour le cache et les tâches asynchrones
- **PostgreSQL** pour la base de données en production
- **Nginx** comme reverse proxy
- **Gunicorn** comme serveur WSGI

## 👥 Équipe de développement

- **Nicolas Cassonnet** ([@NicoCasso](https://github.com/NicoCasso)) - Modèles IA et classification
- **Wael Bensoltana** ([@wbensolt](https://github.com/wbensolt)) - Pipeline NLP et optimisations
- **Dorothée Catry** ([@DorotheeCatry](https://github.com/DorotheeCatry)) - Architecture Django et interface

## 📄 Licence

MIT License - Voir le fichier [LICENSE](LICENSE) pour plus de détails.

## 🤝 Contribution

Les contributions sont les bienvenues ! Merci de :

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/AmazingFeature`)
3. Commit vos changements (`git commit -m 'Add some AmazingFeature'`)
4. Push vers la branche (`git push origin feature/AmazingFeature`)
5. Ouvrir une Pull Request

## 📞 Support

Pour toute question ou problème :
- Ouvrir une [issue GitHub](https://github.com/DorotheeCatry/aura/issues)
- Consulter la [documentation](https://github.com/DorotheeCatry/aura/wiki)
- Contacter l'équipe de développement

---

**AURA** - Assistant Médical IA pour l'hôpital du futur 🏥✨