#!/usr/bin/env python3
"""
Script de diagnostic pour vérifier la compatibilité avec HiTZ/Medical-mT5-large
"""

import sys
import os
import platform
import psutil
import subprocess
import importlib
from pathlib import Path

def print_header(title):
    """Affiche un titre formaté"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)

def print_status(check, status, details=""):
    """Affiche le statut d'une vérification"""
    symbol = "✅" if status else "❌"
    print(f"{symbol} {check}: {details}")

def check_python_version():
    """Vérifie la version de Python"""
    print_header("VERSION PYTHON")
    version = sys.version_info
    is_compatible = version.major == 3 and version.minor >= 8
    print_status("Python version", is_compatible, 
                f"{version.major}.{version.minor}.{version.micro}")
    if not is_compatible:
        print("   ⚠️  Python 3.8+ recommandé pour Transformers")
    return is_compatible

def check_system_resources():
    """Vérifie les ressources système"""
    print_header("RESSOURCES SYSTÈME")
    
    # RAM
    memory = psutil.virtual_memory()
    ram_gb = memory.total / (1024**3)
    ram_available_gb = memory.available / (1024**3)
    ram_ok = ram_gb >= 8
    
    print_status("RAM totale", ram_ok, f"{ram_gb:.1f} GB")
    print_status("RAM disponible", ram_available_gb >= 4, f"{ram_available_gb:.1f} GB")
    
    # Espace disque
    disk = psutil.disk_usage('/')
    disk_free_gb = disk.free / (1024**3)
    disk_ok = disk_free_gb >= 5
    
    print_status("Espace disque libre", disk_ok, f"{disk_free_gb:.1f} GB")
    
    # CPU
    cpu_count = psutil.cpu_count()
    cpu_ok = cpu_count >= 2
    print_status("Processeurs", cpu_ok, f"{cpu_count} cores")
    
    # Système d'exploitation
    print_status("OS", True, f"{platform.system()} {platform.release()}")
    
    return ram_ok and disk_ok and cpu_ok

def check_required_packages():
    """Vérifie les packages Python requis"""
    print_header("PACKAGES PYTHON")
    
    required_packages = {
        'torch': '2.0.0',
        'transformers': '4.21.0',
        'tokenizers': '0.13.0',
        'sentencepiece': '0.1.96',
        'protobuf': 'sourc',
        'numpy': '1.21.0'
    }
    
    all_ok = True
    
    for package, min_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print_status(f"{package}", True, f"v{version}")
        except ImportError:
            print_status(f"{package}", False, "NON INSTALLÉ")
            all_ok = False
    
    return all_ok

def check_huggingface_cache():
    """Vérifie le cache Hugging Face"""
    print_header("CACHE HUGGING FACE")
    
    # Répertoire de cache par défaut
    cache_dir = Path.home() / '.cache' / 'huggingface'
    transformers_cache = cache_dir / 'hub'
    
    print(f"📁 Répertoire cache: {cache_dir}")
    print_status("Cache existe", cache_dir.exists())
    
    if cache_dir.exists():
        # Calculer la taille du cache
        total_size = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
        cache_size_gb = total_size / (1024**3)
        print_status("Taille du cache", True, f"{cache_size_gb:.2f} GB")
        
        # Vérifier si le modèle est déjà téléchargé
        model_cache = transformers_cache / "models--HiTZ--Medical-mT5-large"
        model_cached = model_cache.exists()
        print_status("Medical-mT5-large en cache", model_cached)
        
        if model_cached:
            model_size = sum(f.stat().st_size for f in model_cache.rglob('*') if f.is_file())
            model_size_gb = model_size / (1024**3)
            print(f"   📦 Taille du modèle: {model_size_gb:.2f} GB")
    
    return True

def test_model_loading():
    """Test de chargement du modèle (optionnel)"""
    print_header("TEST DE CHARGEMENT (OPTIONNEL)")
    
    response = input("Voulez-vous tester le chargement du modèle ? (y/N): ").lower().strip()
    
    if response not in ['y', 'yes', 'oui']:
        print("⏩ Test de chargement ignoré")
        return True
    
    print("🔄 Tentative de chargement du modèle...")
    print("   (Cela peut prendre plusieurs minutes la première fois)")
    
    try:
        from transformers import AutoTokenizer, MT5ForConditionalGeneration
        
        model_name = "HiTZ/Medical-mT5-large"
        
        # Test tokenizer
        print("   📝 Chargement du tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print_status("Tokenizer", True, "Chargé avec succès")
        
        # Test modèle
        print("   🧠 Chargement du modèle...")
        model = MT5ForConditionalGeneration.from_pretrained(model_name)
        print_status("Modèle", True, "Chargé avec succès")
        
        # Test simple
        print("   🧪 Test de génération...")
        inputs = tokenizer("question: Qu'est-ce que la grippe ?", return_tensors="pt")
        outputs = model.generate(**inputs, max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print_status("Génération de texte", True, f"'{response[:50]}...'")
        
        return True
        
    except Exception as e:
        print_status("Test de chargement", False, str(e))
        return False

def check_internet_connection():
    """Vérifie la connexion internet"""
    print_header("CONNEXION INTERNET")
    
    try:
        import urllib.request
        urllib.request.urlopen('https://huggingface.co', timeout=10)
        print_status("Connexion Hugging Face", True, "OK")
        return True
    except Exception as e:
        print_status("Connexion Hugging Face", False, str(e))
        return False

def generate_recommendations(results):
    """Génère des recommandations basées sur les résultats"""
    print_header("RECOMMANDATIONS")
    
    if not results['python']:
        print("🔧 Mettez à jour Python vers la version 3.8+")
    
    if not results['resources']:
        print("🔧 Considérez:")
        print("   - Fermer d'autres applications pour libérer de la RAM")
        print("   - Libérer de l'espace disque")
        print("   - Utiliser un modèle plus petit si les ressources sont limitées")
    
    if not results['packages']:
        print("🔧 Installez les packages manquants:")
        print("   pip install -r requirements.txt")
    
    if not results['internet']:
        print("🔧 Vérifiez votre connexion internet pour télécharger les modèles")
    
    if all(results.values()):
        print("🎉 Votre système est prêt pour Medical-mT5-large !")
        print("💡 Vous pouvez maintenant utiliser le modèle en toute confiance.")

def main():
    """Fonction principale"""
    print("🔍 DIAGNOSTIC SYSTÈME POUR MEDICAL-MT5-LARGE")
    print("Ce script vérifie si votre machine peut exécuter le modèle")
    
    # Exécuter toutes les vérifications
    results = {
        'python': check_python_version(),
        'resources': check_system_resources(),
        'packages': check_required_packages(),
        'cache': check_huggingface_cache(),
        'internet': check_internet_connection()
    }
    
    # Test optionnel du modèle
    results['model_test'] = test_model_loading()
    
    # Résumé
    print_header("RÉSUMÉ")
    total_checks = len([k for k in results.keys() if k != 'model_test'])
    passed_checks = sum([1 for k, v in results.items() if v and k != 'model_test'])
    
    print(f"📊 Vérifications réussies: {passed_checks}/{total_checks}")
    
    if passed_checks == total_checks:
        print("✅ Votre système est compatible !")
    else:
        print("⚠️  Quelques ajustements sont nécessaires")
    
    # Générer les recommandations
    generate_recommendations(results)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Diagnostic interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur durant le diagnostic: {e}")
        print("Veuillez vérifier que toutes les dépendances sont installées")