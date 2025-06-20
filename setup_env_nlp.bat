@echo off
echo ==========================================
echo 📦 Setup Environnement NLP avec Poetry
echo ==========================================

REM Étape 1 : Initialisation du projet si pyproject.toml n'existe pas
IF NOT EXIST "pyproject.toml" (
    echo Initialisation du projet...
    poetry init --name aura-clinical-nlp --no-interaction
)

REM Étape 2 : Définir la version de Python dans pyproject.toml
REM Ajout automatique si besoin
powershell -Command "(Get-Content pyproject.toml) -replace '\[tool.poetry.dependencies\]', '[tool.poetry.dependencies]`npython = \"^3.11\"' | Set-Content pyproject.toml"

REM Étape 3 : Installation des dépendances
poetry add transformers datasets torchaudio librosa sentencepiece gradio scikit-learn huggingface_hub jupyter ipykernel

REM Étape 4 : Créer le kernel Jupyter
poetry run python -m ipykernel install --user --name=aura-nlp --display-name "Python (aura-nlp)"

echo ==========================================
echo ✅ Environnement Poetry prêt à l'emploi
echo 📓 Tu peux maintenant lancer : poetry run jupyter notebook
echo ==========================================
pause
