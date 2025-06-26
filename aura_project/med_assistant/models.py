from django.db import models
from django.core.validators import FileExtensionValidator
import json


class Patient(models.Model):
    """Modèle représentant un patient"""
    nom = models.CharField(max_length=100, verbose_name="Nom")
    prenom = models.CharField(max_length=100, verbose_name="Prénom")
    date_naissance = models.DateField(verbose_name="Date de naissance")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Patient"
        verbose_name_plural = "Patients"
        ordering = ['nom', 'prenom']

    def __str__(self):
        return f"{self.nom} {self.prenom}"

    @property
    def nom_complet(self):
        return f"{self.prenom} {self.nom}"

    @property
    def age(self):
        from datetime import date
        today = date.today()
        return today.year - self.date_naissance.year - ((today.month, today.day) < (self.date_naissance.month, self.date_naissance.day))

    @property
    def pathologies_principales(self):
        """Retourne les pathologies principales du patient basées sur ses observations"""
        from collections import Counter
        
        # Récupérer toutes les observations avec classification
        observations = self.observations.filter(
            theme_classe__isnull=False,
            traitement_termine=True
        ).values_list('theme_classe', flat=True)
        
        if not observations:
            return []
        
        # Compter les occurrences
        counter = Counter(observations)
        
        # Retourner les pathologies avec leur nom d'affichage et fréquence
        pathologies = []
        for theme_code, count in counter.most_common():
            theme_display = dict(Observation.THEME_CHOICES).get(theme_code, theme_code)
            pathologies.append({
                'code': theme_code,
                'nom': theme_display,
                'count': count,
                'pourcentage': round((count / len(observations)) * 100, 1)
            })
        
        return pathologies

    @property
    def pathologie_dominante(self):
        """Retourne la pathologie la plus fréquente du patient"""
        pathologies = self.pathologies_principales
        return pathologies[0] if pathologies else None

    @property
    def derniere_observation_date(self):
        """Retourne la date de la dernière observation"""
        derniere = self.observations.order_by('-date').first()
        return derniere.date if derniere else None

    @property
    def observations_terminees_count(self):
        """Nombre d'observations terminées avec succès"""
        return self.observations.filter(traitement_termine=True).count()


class Observation(models.Model):
    """Modèle représentant une observation médicale avec traitement NLP"""
    
    # Mapping des pathologies du modèle IA
    PATHOLOGY_MAPPING = {
        0: 'cardiovasculaire',
        1: 'psy',
        2: 'diabete'
    }
    
    # Choix de thèmes étendus (incluant les pathologies du modèle + autres)
    THEME_CHOICES = [
        ('cardiovasculaire', 'Cardiovasculaire'),
        ('psy', 'Psychique/Neuropsychiatrique'),
        ('diabete', 'Métabolique/Diabète'),
        ('neuro', 'Neurologie'),
        ('pneumo', 'Pneumologie'),
        ('gastro', 'Gastroentérologie'),
        ('ortho', 'Orthopédie'),
        ('dermato', 'Dermatologie'),
        ('general', 'Médecine générale'),
        ('autre', 'Autre'),
    ]

    # Nouvelles catégories d'entités médicales avec noms compréhensibles
    ENTITY_CATEGORIES = [
        ('Maladies et Symptômes', 'Maladies et Symptômes'),
        ('Médicaments', 'Médicaments'),
        ('Anatomie', 'Anatomie'),
        ('Procédures Médicales', 'Procédures Médicales'),
    ]

    patient = models.ForeignKey(
        Patient, 
        on_delete=models.CASCADE, 
        related_name='observations',
        verbose_name="Patient"
    )
    date = models.DateTimeField(auto_now_add=True, verbose_name="Date de création")
    
    # Données d'entrée
    texte_saisi = models.TextField(
        blank=True, 
        null=True, 
        verbose_name="Texte saisi",
        help_text="Note médicale saisie manuellement"
    )
    audio_file = models.FileField(
        upload_to='observations/audio/',
        blank=True,
        null=True,
        validators=[FileExtensionValidator(allowed_extensions=['mp3', 'wav', 'ogg', 'm4a', 'webm'])],
        verbose_name="Fichier audio",
        help_text="Fichier audio de la note médicale"
    )
    
    # Résultats du traitement NLP
    transcription = models.TextField(
        blank=True, 
        null=True, 
        verbose_name="Transcription",
        help_text="Transcription automatique de l'audio"
    )
    theme_classe = models.CharField(
        max_length=20,
        choices=THEME_CHOICES,
        blank=True,
        null=True,
        verbose_name="Thème classé",
        help_text="Classification automatique du thème médical"
    )
    
    # Nouveau champ pour stocker la prédiction numérique du modèle
    model_prediction = models.IntegerField(
        blank=True,
        null=True,
        verbose_name="Prédiction du modèle",
        help_text="Classe numérique prédite par le modèle IA (0, 1, 2)"
    )
    
    resume = models.TextField(
        blank=True, 
        null=True, 
        verbose_name="Résumé",
        help_text="Résumé automatique généré"
    )
    entites = models.JSONField(
        default=dict,
        blank=True,
        verbose_name="Entités extraites",
        help_text="Entités médicales extraites automatiquement"
    )
    
    # Métadonnées de traitement
    traitement_termine = models.BooleanField(
        default=False,
        verbose_name="Traitement terminé"
    )
    traitement_erreur = models.TextField(
        blank=True,
        null=True,
        verbose_name="Erreur de traitement"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Observation"
        verbose_name_plural = "Observations"
        ordering = ['-date']

    def __str__(self):
        return f"Observation {self.patient.nom_complet} - {self.date.strftime('%d/%m/%Y %H:%M')}"

    @property
    def texte_source(self):
        """Retourne le texte source (saisi ou transcrit)"""
        return self.transcription if self.transcription else self.texte_saisi

    @property
    def entites_formatees(self):
        """Retourne les entités sous forme lisible"""
        if not self.entites:
            return {}
        return self.entites

    @classmethod
    def get_pathology_from_prediction(cls, prediction):
        """Convertit une prédiction numérique en thème de pathologie"""
        return cls.PATHOLOGY_MAPPING.get(prediction, 'autre')

    @classmethod
    def get_pathology_display_name(cls, prediction):
        """Retourne le nom d'affichage de la pathologie"""
        theme_code = cls.get_pathology_from_prediction(prediction)
        theme_dict = dict(cls.THEME_CHOICES)
        return theme_dict.get(theme_code, 'Autre')

    def set_prediction_result(self, prediction):
        """Définit le résultat de prédiction et met à jour le thème"""
        self.model_prediction = prediction
        self.theme_classe = self.get_pathology_from_prediction(prediction)

    def get_theme_display_color(self):
        """Retourne la couleur associée au thème"""
        colors = {
            'cardiovasculaire': 'bg-red-50 text-red-700 border-red-200',
            'psy': 'bg-purple-50 text-purple-700 border-purple-200',
            'diabete': 'bg-yellow-50 text-yellow-700 border-yellow-200',
            'neuro': 'bg-indigo-50 text-indigo-700 border-indigo-200',
            'pneumo': 'bg-blue-50 text-blue-700 border-blue-200',
            'gastro': 'bg-green-50 text-green-700 border-green-200',
            'ortho': 'bg-gray-50 text-gray-700 border-gray-200',
            'dermato': 'bg-pink-50 text-pink-700 border-pink-200',
            'general': 'bg-slate-50 text-slate-700 border-slate-200',
            'autre': 'bg-neutral-50 text-neutral-700 border-neutral-200',
        }
        return colors.get(self.theme_classe, 'bg-gray-50 text-gray-700 border-gray-200')

    def get_entity_color(self, entity_type):
        """Retourne la couleur pour un type d'entité avec noms compréhensibles"""
        colors = {
            'Maladies et Symptômes': 'bg-red-50 text-red-700 border-red-200',
            'Médicaments': 'bg-blue-50 text-blue-700 border-blue-200',
            'Anatomie': 'bg-green-50 text-green-700 border-green-200',
            'Procédures Médicales': 'bg-purple-50 text-purple-700 border-purple-200',
        }
        return colors.get(entity_type, 'bg-gray-50 text-gray-700 border-gray-200')