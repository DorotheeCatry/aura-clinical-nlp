"""
Client API pour communiquer avec l'API FastAPI
"""

import requests
import logging
from typing import Dict, List, Optional
from django.conf import settings

logger = logging.getLogger(__name__)


class FastAPIClient:
    """Client pour communiquer avec l'API FastAPI"""
    
    def __init__(self):
        # URL de base de l'API FastAPI (à configurer dans settings)
        self.base_url = getattr(settings, 'FASTAPI_BASE_URL', 'http://127.0.0.1:8000')
        self.timeout = getattr(settings, 'FASTAPI_TIMEOUT', 30)
        
    def get_available_models(self) -> List[str]:
        """
        Récupère la liste des modèles disponibles
        
        Returns:
            Liste des noms de modèles disponibles
        """
        try:
            response = requests.get(
                f"{self.base_url}/get_available_models",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return data.get('model_names', [])
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors de la récupération des modèles: {e}")
            return []
        except Exception as e:
            logger.error(f"Erreur inattendue: {e}")
            return []
    
    def process_text(self, model_name: str, text: str) -> Dict[str, str]:
        """
        Traite un texte avec le modèle spécifié
        
        Args:
            model_name: Nom du modèle à utiliser
            text: Texte à traiter
            
        Returns:
            Dictionnaire avec la réponse du modèle
        """
        try:
            payload = {
                "requested_model": model_name,
                "question_text": text
            }
            
            response = requests.post(
                f"{self.base_url}/process_text",
                json=payload,
                timeout=self.timeout,
                headers={'Content-Type': 'application/json'}
            )
            response.raise_for_status()
            
            data = response.json()
            return {
                'model_name': data.get('model_name', 'unknown'),
                'response': data.get('response', 'Erreur de traitement'),
                'success': True,
                'error': None
            }
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Erreur lors du traitement du texte: {e}")
            return {
                'model_name': model_name,
                'response': '',
                'success': False,
                'error': f"Erreur de connexion: {e}"
            }
        except Exception as e:
            logger.error(f"Erreur inattendue: {e}")
            return {
                'model_name': model_name,
                'response': '',
                'success': False,
                'error': f"Erreur inattendue: {e}"
            }
    
    def is_api_available(self) -> bool:
        """
        Vérifie si l'API FastAPI est disponible
        
        Returns:
            True si l'API est disponible, False sinon
        """
        try:
            response = requests.get(
                f"{self.base_url}/get_available_models",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False


# Instance globale du client
fastapi_client = FastAPIClient()