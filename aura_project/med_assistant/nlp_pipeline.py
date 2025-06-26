"""
Pipeline NLP pour AURA - Assistant Médical
Traitement automatique des observations médicales avec modèles Hugging Face directs
Optimisé pour GPU avec mémoire limitée
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
import json
import os
import tempfile

# Imports pour Whisper (transcription)
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

# Imports pour les modèles Hugging Face
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification,
        T5Tokenizer, T5ForConditionalGeneration, pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"⚠️ Transformers non disponible: {e}")

logger = logging.getLogger(__name__)


class NLPPipeline:
    """
    Pipeline de traitement NLP pour les observations médicales
    Intègre : transcription Whisper, classification, extraction d'entités DrBERT-CASM2, résumé T5
    Optimisé pour GPU avec mémoire limitée
    """
    
    def __init__(self):
        """Initialise la pipeline NLP avec les modèles Hugging Face directs"""
        self.models_loaded = False
        self.whisper_available = WHISPER_AVAILABLE
        self.transformers_available = TRANSFORMERS_AVAILABLE
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if WHISPER_AVAILABLE else "cpu"
        
        # Modèles chargés à la demande pour économiser la mémoire
        self.whisper_model = None
        self.whisper_processor = None
        self.classification_model = None
        self.classification_tokenizer = None
        self.drbert_pipeline = None
        self.t5_pipeline = None
        
        # Configuration des modèles - NOUVEAU DrBERT-CASM2
        self.models_config = {
            'classification': 'waelbensoltana/finetuned-medical-fr',
            'entities': 'medkit/DrBERT-CASM2',  # NOUVEAU MODÈLE !
            'summarization': 'plguillou/t5-base-fr-sum-cnndm'
        }
        
        # Mapping des pathologies du modèle
        self.pathology_mapping = {
            0: 'cardiovasculaire',
            1: 'psy', 
            2: 'diabete'
        }
        
        # Mapping des entités DrBERT-CASM2 vers nos catégories
        self.drbert_entity_mapping = {
            'DISO': 'DISO',  # Disorders/Maladies
            'CHEM': 'CHEM',  # Chemicals/Médicaments
            'ANAT': 'ANAT',  # Anatomie
            'PROC': 'PROC',  # Procédures
            'LIVB': 'ANAT',  # Living beings -> Anatomie
            'OBJC': 'PROC',  # Objects -> Procédures
            'PHEN': 'DISO',  # Phenomena -> Disorders
            'PHYS': 'DISO',  # Physiology -> Disorders
            'GEOG': 'PROC',  # Geography -> Procédures
            'CONC': 'PROC',  # Concepts -> Procédures
        }
        
        self._load_models()
    
    def _clear_gpu_cache(self):
        """Nettoie le cache GPU pour libérer de la mémoire"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 Cache GPU nettoyé")
    
    def _load_models(self):
        """
        Charge les modèles NLP incluant Whisper
        Optimisé pour GPU avec mémoire limitée
        """
        try:
            # Charger Whisper pour la transcription (local) - PRIORITÉ
            if self.whisper_available:
                logger.info("🎤 Chargement du modèle Whisper...")
                
                self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
                self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
                self.whisper_model.to(self.device)
                
                logger.info(f"✅ Whisper chargé sur {self.device}")
                
                # Afficher l'utilisation mémoire après Whisper
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated() / 1024**3
                    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    logger.info(f"📊 Mémoire GPU après Whisper: {memory_used:.2f}GB / {memory_total:.2f}GB")
            else:
                logger.warning("⚠️ Whisper non disponible")
            
            # Les autres modèles seront chargés à la demande pour économiser la mémoire
            logger.info("💡 Classification, DrBERT-CASM2 et T5 seront chargés à la demande pour optimiser la mémoire")
            
            self.models_loaded = True
            logger.info("✅ Pipeline NLP initialisée avec succès")
            
        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
            self.models_loaded = False
            self.whisper_available = False
            self.transformers_available = False
    
    def _load_classification_on_demand(self):
        """Charge le modèle de classification à la demande"""
        if self.classification_model is not None:
            return True
            
        if not self.transformers_available:
            return False
            
        try:
            logger.info("🏷️ Chargement du modèle de classification à la demande...")
            
            self.classification_tokenizer = AutoTokenizer.from_pretrained(self.models_config['classification'])
            self.classification_model = AutoModelForSequenceClassification.from_pretrained(
                self.models_config['classification'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            if torch.cuda.is_available():
                self.classification_model.to(self.device)
            
            logger.info(f"✅ Modèle de classification chargé sur {self.device}")
            
            # Afficher l'utilisation mémoire
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"📊 Mémoire GPU après classification: {memory_used:.2f}GB / {memory_total:.2f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèle de classification: {e}")
            self.transformers_available = False
            return False
    
    def _load_drbert_on_demand(self):
        """Charge DrBERT-CASM2 à la demande et libère le modèle de classification si nécessaire"""
        if self.drbert_pipeline is not None:
            return True
            
        if not self.transformers_available:
            return False
            
        try:
            logger.info("🧠 Chargement du modèle DrBERT-CASM2 à la demande...")
            
            # Libérer le modèle de classification temporairement si nécessaire
            classification_was_loaded = self.classification_model is not None
            if classification_was_loaded and torch.cuda.is_available():
                logger.info("🔄 Libération temporaire du modèle de classification pour DrBERT-CASM2...")
                del self.classification_model
                del self.classification_tokenizer
                self.classification_model = None
                self.classification_tokenizer = None
                self._clear_gpu_cache()
            
            # Charger DrBERT-CASM2 avec optimisations mémoire
            tokenizer = AutoTokenizer.from_pretrained(self.models_config['entities'])
            model = AutoModelForTokenClassification.from_pretrained(
                self.models_config['entities'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Créer le pipeline NER avec aggregation_strategy="simple" comme dans votre exemple
            self.drbert_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info(f"✅ DrBERT-CASM2 chargé sur {self.device}")
            
            # Afficher l'utilisation mémoire
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"📊 Mémoire GPU après DrBERT-CASM2: {memory_used:.2f}GB / {memory_total:.2f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement DrBERT-CASM2: {e}")
            self.transformers_available = False
            
            # Recharger le modèle de classification si il était chargé
            if classification_was_loaded:
                self._load_classification_on_demand()
            
            return False
    
    def _load_t5_on_demand(self):
        """Charge T5 à la demande et libère d'autres modèles si nécessaire"""
        if self.t5_pipeline is not None:
            return True
            
        if not self.transformers_available:
            return False
            
        try:
            logger.info("📝 Chargement du modèle T5 à la demande...")
            
            # Libérer DrBERT temporairement si nécessaire
            drbert_was_loaded = self.drbert_pipeline is not None
            if drbert_was_loaded:
                logger.info("🔄 Libération temporaire de DrBERT-CASM2 pour T5...")
                del self.drbert_pipeline
                self.drbert_pipeline = None
                self._clear_gpu_cache()
            
            # Charger T5 avec optimisations mémoire
            tokenizer = T5Tokenizer.from_pretrained(self.models_config['summarization'], legacy=False)
            model = T5ForConditionalGeneration.from_pretrained(
                self.models_config['summarization'],
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            
            # Créer le pipeline de résumé avec optimisations
            self.t5_pipeline = pipeline(
                "summarization",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info(f"✅ T5 résumé chargé sur {self.device}")
            
            # Afficher l'utilisation mémoire
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / 1024**3
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"📊 Mémoire GPU après T5: {memory_used:.2f}GB / {memory_total:.2f}GB")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement T5: {e}")
            self.transformers_available = False
            return False
    
    def _reload_whisper(self):
        """Recharge Whisper si nécessaire"""
        if self.whisper_model is not None or not self.whisper_available:
            return
            
        try:
            logger.info("🔄 Rechargement de Whisper...")
            self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
            self.whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
            self.whisper_model.to(self.device)
            logger.info("✅ Whisper rechargé")
        except Exception as e:
            logger.error(f"❌ Erreur rechargement Whisper: {e}")
    
    def clean_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Votre fonction clean_entities exactement comme vous l'avez écrite !
        Post-traitement pour nettoyer les entités DrBERT-CASM2
        
        Args:
            entities: Liste des entités brutes de DrBERT-CASM2
            
        Returns:
            Liste des entités nettoyées
        """
        cleaned = []
        for ent in entities:
            word = ent["word"].replace("##", "")
            # On fusionne si c'est la suite d'un mot précédent
            if cleaned and ent["start"] == cleaned[-1]["end"]:
                cleaned[-1]["word"] += word
                cleaned[-1]["end"] = ent["end"]
            else:
                cleaned.append({
                    "entity_group": ent["entity_group"],
                    "word": word,
                    "start": ent["start"],
                    "end": ent["end"],
                    "score": ent["score"]
                })
        return cleaned
    
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
                logger.warning("⚠️ Whisper non disponible")
                return None
            
            # S'assurer que Whisper est chargé
            if self.whisper_model is None:
                self._reload_whisper()
                if self.whisper_model is None:
                    return None
            
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
            return None
    
    def classify_theme(self, text: str) -> tuple[Optional[str], Optional[int]]:
        """
        Classifie le thème médical avec le modèle waelbensoltana/finetuned-medical-fr
        
        Args:
            text: Texte à classifier
            
        Returns:
            Tuple (thème_classifié, prédiction_numérique) ou (None, None) en cas d'erreur
        """
        try:
            # Charger le modèle de classification à la demande
            if not self._load_classification_on_demand():
                logger.warning("⚠️ Modèle de classification non disponible")
                return None, None
            
            logger.info(f"🏷️ Classification du texte: {text[:50]}...")
            
            # Tokeniser le texte
            inputs = self.classification_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True,
                max_length=512
            )
            
            # Déplacer sur le bon device si nécessaire
            if torch.cuda.is_available():
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Prédiction
            with torch.no_grad():
                outputs = self.classification_model(**inputs)
                prediction = torch.argmax(outputs.logits, dim=1).item()
            
            # Convertir la prédiction en thème
            theme = self.pathology_mapping.get(prediction, 'autre')
            
            logger.info(f"✅ Classification: prédiction={prediction}, thème={theme}")
            
            # Libérer le modèle de classification après utilisation pour économiser la mémoire
            if self.classification_model is not None:
                logger.info("🔄 Libération du modèle de classification après utilisation")
                del self.classification_model
                del self.classification_tokenizer
                self.classification_model = None
                self.classification_tokenizer = None
                self._clear_gpu_cache()
            
            return theme, prediction
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la classification: {e}")
            return None, None
    
    def extract_entities_drbert(self, text: str) -> Dict[str, List[str]]:
        """
        Extrait les entités médicales avec DrBERT-CASM2 en utilisant votre fonction clean_entities
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire des entités extraites par catégorie
        """
        try:
            # Charger DrBERT-CASM2 à la demande
            if not self._load_drbert_on_demand():
                logger.warning("⚠️ DrBERT-CASM2 non disponible")
                return {}
            
            logger.info(f"🔍 Extraction d'entités DrBERT-CASM2 pour: {text[:50]}...")
            
            # Utiliser le pipeline NER de DrBERT-CASM2 avec aggregation_strategy="simple"
            entities = self.drbert_pipeline(text)
            
            logger.info(f"🔍 DrBERT-CASM2 a trouvé {len(entities)} entités brutes")
            
            # DEBUG: Afficher quelques entités brutes
            for i, ent in enumerate(entities[:5]):
                logger.info(f"  Entité brute {i}: {ent}")
            
            # Appliquer votre fonction clean_entities
            cleaned_entities = self.clean_entities(entities)
            
            logger.info(f"🧹 Après clean_entities: {len(cleaned_entities)} entités nettoyées")
            
            # DEBUG: Afficher quelques entités nettoyées
            for i, ent in enumerate(cleaned_entities[:5]):
                logger.info(f"  Entité nettoyée {i}: {ent}")
            
            # Organiser les entités par catégorie
            categorized_entities = {
                'DISO': [],  # Disorders
                'CHEM': [],  # Chemicals/Drugs
                'ANAT': [],  # Anatomy
                'PROC': [],  # Procedures
            }
            
            for entity in cleaned_entities:
                entity_label = entity['entity_group']
                entity_text = entity['word'].strip()
                confidence = entity['score']
                
                # Mapper vers nos catégories
                mapped_category = self.drbert_entity_mapping.get(entity_label, 'PROC')
                
                # Filtrer par confiance (seuil à 0.5) et longueur minimale
                if confidence > 0.5 and len(entity_text) > 2:
                    # Éviter les doublons
                    if entity_text not in categorized_entities[mapped_category]:
                        categorized_entities[mapped_category].append(entity_text)
                        logger.debug(f"  ✓ {mapped_category}: {entity_text} (conf: {confidence:.2f})")
            
            # Nettoyer les catégories vides
            categorized_entities = {k: v for k, v in categorized_entities.items() if v}
            
            logger.info(f"✅ DrBERT-CASM2: {sum(len(v) for v in categorized_entities.values())} entités extraites et nettoyées")
            
            # DEBUG: Afficher le résultat final
            for category, entities_list in categorized_entities.items():
                logger.info(f"  {category}: {entities_list}")
            
            # Libérer DrBERT après utilisation pour économiser la mémoire
            if self.drbert_pipeline is not None:
                logger.info("🔄 Libération de DrBERT-CASM2 après utilisation")
                del self.drbert_pipeline
                self.drbert_pipeline = None
                self._clear_gpu_cache()
            
            return categorized_entities
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de l'extraction DrBERT-CASM2: {e}")
            logger.exception("Détails de l'erreur:")
            return {}
    
    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extrait les entités médicales (utilise DrBERT-CASM2 maintenant)
        
        Args:
            text: Texte à analyser
            
        Returns:
            Dictionnaire des entités extraites
        """
        return self.extract_entities_drbert(text)
    
    def generate_summary_t5(self, text: str) -> Optional[str]:
        """
        Génère un résumé avec T5 français (chargé à la demande)
        
        Args:
            text: Texte à résumer
            
        Returns:
            Résumé généré ou None en cas d'erreur
        """
        try:
            # Charger T5 à la demande
            if not self._load_t5_on_demand():
                logger.warning("⚠️ T5 non disponible")
                return None
            
            logger.info(f"📝 Génération résumé T5 pour: {text[:50]}...")
            
            # Utiliser le pipeline de résumé T5
            summary = self.t5_pipeline(
                text, 
                max_length=100, 
                min_length=20, 
                do_sample=False
            )[0]['summary_text']
            
            logger.info(f"✅ T5: Résumé généré ({len(summary)} caractères)")
            
            # Libérer T5 après utilisation pour économiser la mémoire
            if self.t5_pipeline is not None:
                logger.info("🔄 Libération de T5 après utilisation")
                del self.t5_pipeline
                self.t5_pipeline = None
                self._clear_gpu_cache()
            
            return summary.strip()
            
        except Exception as e:
            logger.error(f"❌ Erreur lors de la génération T5: {e}")
            return None
    
    def generate_summary(self, text: str) -> Optional[str]:
        """
        Génère un résumé via T5 local
        
        Args:
            text: Texte à résumer
            
        Returns:
            Résumé généré ou None en cas d'erreur
        """
        return self.generate_summary_t5(text)
    
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
            'model_prediction': None,
            'resume': None,
            'entites': {},
            'success': False,
            'error': None,
            'classification_used': self.transformers_available,
            'drbert_used': self.transformers_available,
            't5_used': self.transformers_available
        }
        
        try:
            logger.info(f"🔄 Début traitement observation {observation.id}")
            
            # 1. Transcription si fichier audio (toujours local avec Whisper)
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
            
            # 3. Classification du thème avec prédiction numérique
            theme, prediction = self.classify_theme(text_source)
            if theme:
                results['theme_classe'] = theme
                results['model_prediction'] = prediction
                logger.info(f"🏷️ Thème classifié: {theme} (prédiction: {prediction})")
            
            # 4. Extraction d'entités (DrBERT-CASM2 avec votre fonction clean_entities)
            entities = self.extract_entities(text_source)
            results['entites'] = entities
            logger.info(f"🔍 Entités extraites avec clean_entities: {len(entities)} catégories")
            
            # 5. Génération du résumé (T5 local à la demande)
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
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retourne le statut de la pipeline
        
        Returns:
            Dictionnaire avec le statut des différents composants
        """
        return {
            'whisper_available': self.whisper_available,
            'drbert_available': self.transformers_available,
            't5_available': self.transformers_available,
            'classification_available': self.transformers_available,
            'available_models': [
                'waelbensoltana/finetuned-medical-fr',
                'medkit/DrBERT-CASM2',  # NOUVEAU MODÈLE !
                'plguillou/t5-base-fr-sum-cnndm'
            ] if self.transformers_available else [],
            'models_loaded': self.models_loaded,
            'device': self.device,
            'pathology_mapping': self.pathology_mapping,
            'drbert_entity_mapping': self.drbert_entity_mapping,
            'memory_optimized': True,
            'models_config': self.models_config
        }


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
            observation.model_prediction = results['model_prediction']
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