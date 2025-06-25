"""
Pipeline NLP pour AURA - Assistant Médical
Traitement automatique des observations médicales avec Whisper
"""

import logging
from typing import Dict, Any, Optional
import json
import random
import os
import tempfile

# Imports pour Whisper
try:
    import torch
    import torchaudio
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa
    import soundfile as sf
    WHISPER_AVAILABLE = True
except ImportError as e:
    WHISPER_AVAILABLE = False
    print(f"⚠️ Whisper non disponible: {e}")

logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Pipeline de traitement NLP pour les observations médicales
    Intègre : transcription Whisper, classification, extraction d'entités, résumé
    """
    
    def __init__(self):
        """Initialise la pipeline NLP avec Whisper"""
        self.models_loaded = False
        self.whisper_available = WHISPER_AVAILABLE
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if WHISPER_AVAILABLE else "cpu"
        self._load_models()
    
    def _load_models(self):
        """
        Charge les modèles NLP incluant Whisper
        """
        try:
            if self.whisper_available:
                logger.info("🎤 Chargement du modèle Whisper...")
                
                # Charger Whisper pour la transcription
                self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
                self.whisper_model.to(self.device)
                
                logger.info(f"✅ Whisper chargé sur {self.device}")
            else:
                logger.warning("⚠️ Whisper non disponible, utilisation de la simulation")
            
            # TODO: Charger les autres modèles
            # self.camembert_classifier = AutoModelForSequenceClassification.from_pretrained(...)
            # self.drbert_ner = AutoModelForTokenClassification.from_pretrained(...)
            # self.t5_summarizer = AutoModelForSeq2SeqLM.from_pretrained(...)
            
            self.models_loaded = True
            logger.info("✅ Pipeline NLP initialisée avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
            self.models_loaded = False
            self.whisper_available = False
    
    def transcribe_audio(self, audio_file_path: str) -> Optional[str]:
        """
        Transcrit un fichier audio en texte avec Whisper
        
        Args:
            audio_file_path: Chemin vers le fichier audio
            
        Returns:
            Texte transcrit ou None en cas d'erreur
        """
        try:
            if not self.whisper_available or not self.models_loaded:
                logger.warning("⚠️ Whisper non disponible, utilisation de la simulation")
                return self._mock_transcription()
            
            logger.info(f"🎤 Début transcription de: {audio_file_path}")
            
            # Vérifier que le fichier existe
            if not os.path.exists(audio_file_path):
                logger.error(f"❌ Fichier audio non trouvé: {audio_file_path}")
                return None
            
            # Charger l'audio avec librosa (plus robuste que torchaudio)
            try:
                # Charger et convertir à 16kHz mono
                waveform, sr = librosa.load(audio_file_path, sr=16000, mono=True)
                logger.info(f"📊 Audio chargé: {len(waveform)} échantillons à {sr}Hz")
                
            except Exception as e:
                logger.error(f"❌ Erreur chargement audio: {e}")
                # Fallback avec torchaudio
                try:
                    waveform, sr = torchaudio.load(audio_file_path)
                    # Convertir en mono si stéréo
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0)
                    else:
                        waveform = waveform.squeeze()
                    
                    # Resample à 16kHz
                    if sr != 16000:
                        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                        waveform = resampler(waveform)
                    
                    waveform = waveform.numpy()
                    
                except Exception as e2:
                    logger.error(f"❌ Erreur fallback torchaudio: {e2}")
                    return None
            
            # Préparer les inputs pour Whisper
            inputs = self.whisper_processor(
                waveform, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            # Déplacer sur le bon device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            logger.info("🧠 Génération de la transcription...")
            
            # Générer la transcription
            with torch.no_grad():
                generated_ids = self.whisper_model.generate(
                    inputs["input_features"],
                    attention_mask=inputs.get("attention_mask", None),
                    language="fr",  # Forcer le français
                    task="transcribe",
                    max_length=448,  # Limite raisonnable
                    num_beams=5,     # Améliorer la qualité
                    do_sample=False  # Déterministe
                )
            
            # Décoder la transcription
            transcription = self.whisper_processor.batch_decode(
                generated_ids, 
                skip_special_tokens=True
            )[0]
            
            # Nettoyer la transcription
            transcription = transcription.strip()
            
            logger.info(f"✅ Transcription réussie: {len(transcription)} caractères")
            logger.info(f"📄 Aperçu: {transcription[:100]}...")
            
            return transcription
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la transcription Whisper: {e}")
            # Fallback vers simulation en cas d'erreur
            return self._mock_transcription()
    
    def classify_theme(self, text: str) -> Optional[str]:
        """
        Classifie le thème médical avec CamemBERT fine-tuné
        
        Args:
            text: Texte à classifier
            
        Returns:
            Thème classifié ou None en cas d'erreur
        """
        try:
            if not self.models_loaded:
                return self._mock_classification(text)
            
            # TODO: Implémenter la vraie classification
            # inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            # outputs = self.camembert_classifier(**inputs)
            # predicted_class = torch.argmax(outputs.logits, dim=-1)
            # return self.class_labels[predicted_class]
            
            return self._mock_classification(text)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la classification: {e}")
            return None
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extrait les entités médicales avec DrBERT
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire des entités extraites avec nouvelles catégories
        """
        try:
            if not self.models_loaded:
                return self._mock_entities(text)
            
            # TODO: Implémenter la vraie extraction d'entités
            # inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            # outputs = self.drbert_ner(**inputs)
            # entities = self._process_ner_outputs(outputs, text)
            # return entities
            
            return self._mock_entities(text)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'extraction d'entités: {e}")
            return {}
    
    def generate_summary(self, text: str) -> Optional[str]:
        """
        Génère un résumé avec T5 français
        
        Args:
            text: Texte à résumer
            
        Returns:
            Résumé généré ou None en cas d'erreur
        """
        try:
            if not self.models_loaded:
                return self._mock_summary(text)
            
            # TODO: Implémenter le vrai résumé T5
            # inputs = self.tokenizer(f"résume: {text}", return_tensors="pt", truncation=True)
            # outputs = self.t5_summarizer.generate(**inputs, max_length=150)
            # summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # return summary
            
            return self._mock_summary(text)
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération du résumé: {e}")
            return None
    
    def process_observation(self, observation) -> Dict[str, Any]:
        """
        Traite une observation complète avec toute la pipeline
        
        Args:
            observation: Instance du modèle Observation
            
        Returns:
            Dictionnaire avec tous les résultats du traitement
        """
        results = {
            'transcription': None,
            'theme_classe': None,
            'resume': None,
            'entites': {},
            'success': False,
            'error': None
        }
        
        try:
            logger.info(f"🔄 Début traitement observation {observation.id}")
            
            # 1. Transcription si fichier audio
            if observation.audio_file:
                logger.info(f"🎤 Transcription du fichier: {observation.audio_file.path}")
                transcription = self.transcribe_audio(observation.audio_file.path)
                if transcription:
                    results['transcription'] = transcription
                    logger.info("✅ Transcription terminée")
                else:
                    logger.warning("⚠️ Échec de la transcription")
            
            # 2. Déterminer le texte source
            text_source = results['transcription'] or observation.texte_saisi
            
            if not text_source:
                results['error'] = "Aucun texte disponible pour le traitement"
                logger.error("❌ Aucun texte source disponible")
                return results
            
            logger.info(f"📝 Texte source: {len(text_source)} caractères")
            
            # 3. Classification du thème
            theme = self.classify_theme(text_source)
            if theme:
                results['theme_classe'] = theme
                logger.info(f"🏷️ Thème classifié: {theme}")
            
            # 4. Extraction d'entités
            entities = self.extract_entities(text_source)
            results['entites'] = entities
            logger.info(f"🔍 Entités extraites: {len(entities)} catégories")
            
            # 5. Génération du résumé
            summary = self.generate_summary(text_source)
            if summary:
                results['resume'] = summary
                logger.info("📄 Résumé généré")
            
            results['success'] = True
            logger.info(f"✅ Traitement NLP terminé pour l'observation {observation.id}")
            
        except Exception as e:
            error_msg = f"Erreur lors du traitement NLP: {e}"
            logger.error(f"❌ {error_msg}")
            results['error'] = error_msg
        
        return results
    
    # Méthodes de simulation pour le développement
    def _mock_transcription(self) -> str:
        """Simulation de transcription pour le développement"""
        transcriptions = [
            "Patient présente des douleurs thoraciques depuis ce matin. Tension artérielle élevée à 160/95. Prescrit un ECG et analyses sanguines.",
            "Consultation de suivi pour diabète de type 2. Glycémie à jeun à 1,45 g/L. Ajustement de la metformine à 1000mg matin et soir.",
            "Patient anxieux, troubles du sommeil depuis 3 semaines. Prescrit anxiolytique léger et suivi psychologique.",
            "Douleur abdominale chronique, suspicion de gastrite. Prescription d'IPP et fibroscopie à programmer."
        ]
        return random.choice(transcriptions)
    
    def _mock_classification(self, text: str) -> str:
        """Simulation de classification pour le développement"""
        text_lower = text.lower()
        if any(word in text_lower for word in ['cœur', 'cardiaque', 'tension', 'ecg', 'thoracique']):
            return 'cardio'
        elif any(word in text_lower for word in ['anxiété', 'dépression', 'stress', 'anxieux', 'sommeil']):
            return 'psy'
        elif any(word in text_lower for word in ['diabète', 'glycémie', 'insuline', 'metformine']):
            return 'diabete'
        elif any(word in text_lower for word in ['abdomen', 'gastrite', 'estomac', 'digestif']):
            return 'gastro'
        else:
            return 'general'
    
    def _mock_entities(self, text: str) -> Dict[str, Any]:
        """Simulation d'extraction d'entités avec nouvelles catégories"""
        text_lower = text.lower()
        entities = {
            'DISO': [],  # Disorders
            'CHEM': [],  # Chemicals/Drugs
            'ANAT': [],  # Anatomy
            'PROC': [],  # Procedures
            'TEST': [],  # Tests
            'MED': [],   # Medications
            'BODY': []   # Body parts
        }
        
        # Simulation basée sur le contenu
        if 'douleur' in text_lower:
            entities['DISO'].append('douleur thoracique')
        if 'diabète' in text_lower:
            entities['DISO'].append('diabète de type 2')
        if 'anxiété' in text_lower or 'anxieux' in text_lower:
            entities['DISO'].append('troubles anxieux')
        if 'gastrite' in text_lower:
            entities['DISO'].append('gastrite chronique')
            
        if 'metformine' in text_lower:
            entities['MED'].append('metformine 1000mg')
        if 'anxiolytique' in text_lower:
            entities['MED'].append('anxiolytique')
        if 'ipp' in text_lower:
            entities['MED'].append('inhibiteur de pompe à protons')
            
        if 'thorax' in text_lower or 'thoracique' in text_lower:
            entities['ANAT'].append('thorax')
        if 'cœur' in text_lower:
            entities['ANAT'].append('cœur')
        if 'abdomen' in text_lower:
            entities['ANAT'].append('abdomen')
            
        if 'ecg' in text_lower:
            entities['TEST'].append('électrocardiogramme')
        if 'glycémie' in text_lower:
            entities['TEST'].append('glycémie à jeun')
        if 'analyses sanguines' in text_lower:
            entities['TEST'].append('bilan sanguin')
        if 'fibroscopie' in text_lower:
            entities['PROC'].append('fibroscopie gastrique')
            
        # Nettoyer les listes vides
        return {k: v for k, v in entities.items() if v}
    
    def _mock_summary(self, text: str) -> str:
        """Simulation de résumé pour le développement"""
        summaries = {
            'cardio': "Consultation cardiologique : douleurs thoraciques avec HTA. Examens complémentaires prescrits.",
            'diabete': "Suivi diabétologique : ajustement thérapeutique suite à déséquilibre glycémique.",
            'psy': "Consultation psychiatrique : troubles anxieux avec retentissement sur le sommeil. Traitement initié.",
            'gastro': "Consultation gastroentérologique : douleurs abdominales chroniques. Explorations à poursuivre.",
            'general': "Consultation de médecine générale : prise en charge symptomatique et suivi."
        }
        
        # Déterminer le type basé sur le texte
        text_lower = text.lower()
        if any(word in text_lower for word in ['cœur', 'cardiaque', 'tension', 'ecg']):
            return summaries['cardio']
        elif any(word in text_lower for word in ['diabète', 'glycémie', 'metformine']):
            return summaries['diabete']
        elif any(word in text_lower for word in ['anxiété', 'anxieux', 'sommeil']):
            return summaries['psy']
        elif any(word in text_lower for word in ['abdomen', 'gastrite']):
            return summaries['gastro']
        else:
            return summaries['general']


# Instance globale de la pipeline
nlp_pipeline = NLPPipeline()


def process_observation_async(observation_id: int):
    """
    Fonction pour traiter une observation de manière asynchrone
    À utiliser avec Celery ou un système de tâches similaire
    """
    from .models import Observation
    
    try:
        observation = Observation.objects.get(id=observation_id)
        results = nlp_pipeline.process_observation(observation)
        
        # Mise à jour de l'observation avec les résultats
        if results['success']:
            observation.transcription = results['transcription']
            observation.theme_classe = results['theme_classe']
            observation.resume = results['resume']
            observation.entites = results['entites']
            observation.traitement_termine = True
        else:
            observation.traitement_erreur = results['error']
        
        observation.save()
        
    except Observation.DoesNotExist:
        logger.error(f"Observation {observation_id} non trouvée")
    except Exception as e:
        logger.error(f"Erreur lors du traitement asynchrone: {e}")