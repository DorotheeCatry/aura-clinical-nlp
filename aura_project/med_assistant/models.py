from django.db import models
from django.core.validators import FileExtensionValidator
from django.contrib.auth.models import User
import json


class UserProfile(models.Model):
    """Profil utilisateur étendu avec informations professionnelles"""
    
    ROLE_CHOICES = [
        ('medecin_generaliste', 'Médecin généraliste'),
        ('medecin_specialiste', 'Médecin spécialiste'),
        ('infirmiere', 'Infirmière'),
        ('aide_soignant', 'Aide-soignant'),
        ('sage_femme', 'Sage-femme'),
        ('kinesitherapeute', 'Kinésithérapeute'),
        ('pharmacien', 'Pharmacien'),
        ('psychologue', 'Psychologue'),
        ('administrateur', 'Administrateur'),
        ('autre', 'Autre'),
    ]
    
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    role = models.CharField(
        max_length=30,
        choices=ROLE_CHOICES,
        default='autre',
        verbose_name="Rôle professionnel"
    )
    specialite = models.CharField(
        max_length=100,
        blank=True,
        null=True,
        verbose_name="Spécialité",
        help_text="Spécialité médicale ou domaine d'expertise"
    )
    etablissement = models.CharField(
        max_length=200,
        blank=True,
        null=True,
        verbose_name="Établissement",
        help_text="Hôpital, clinique, cabinet médical..."
    )
    numero_ordre = models.CharField(
        max_length=50,
        blank=True,
        null=True,
        verbose_name="Numéro d'ordre",
        help_text="Numéro RPPS, ADELI ou autre identifiant professionnel"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Profil utilisateur"
        verbose_name_plural = "Profils utilisateurs"

    def __str__(self):
        return f"{self.user.get_full_name()} - {self.get_role_display()}"

    @property
    def display_name(self):
        """Nom d'affichage complet avec titre professionnel"""
        role_display = self.get_role_display()
        full_name = self.user.get_full_name() or self.user.username
        return f"{role_display} {full_name}"


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
    def pathologies(self):
        """Retourne la liste des pathologies uniques du patient"""
        pathologies = set()
        for obs in self.observations.filter(theme_classe__isnull=False):
            if obs.theme_classe:
                pathologies.add(obs.theme_classe)
        return list(pathologies)

    @property
    def pathologies_display(self):
        """Retourne les pathologies avec leurs noms d'affichage"""
        pathologies_dict = dict(Observation.THEME_CHOICES)
        return [pathologies_dict.get(p, p) for p in self.pathologies]

    @property
    def pathologies_count(self):
        """Retourne le nombre de pathologies différentes"""
        return len(self.pathologies)

    @property
    def derniere_observation(self):
        """Retourne la dernière observation du patient"""
        return self.observations.first()


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
    
    # NOUVEAU : Traçabilité de l'auteur
    created_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='observations_created',
        verbose_name="Créé par"
    )
    modified_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='observations_modified',
        verbose_name="Modifié par"
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
        author_info = f" par {self.created_by.get_full_name() or self.created_by.username}" if self.created_by else ""
        return f"Observation {self.patient.nom_complet} - {self.date.strftime('%d/%m/%Y %H:%M')}{author_info}"

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

    @property
    def author_display(self):
        """Affichage de l'auteur avec son rôle professionnel"""
        if not self.created_by:
            return "Utilisateur inconnu"
        
        try:
            profile = self.created_by.profile
            return f"{profile.get_role_display()} {self.created_by.get_full_name() or self.created_by.username}"
        except:
            return self.created_by.get_full_name() or self.created_by.username

    @property
    def modifier_display(self):
        """Affichage du modificateur avec son rôle professionnel"""
        if not self.modified_by:
            return None
        
        try:
            profile = self.modified_by.profile
            return f"{profile.get_role_display()} {self.modified_by.get_full_name() or self.modified_by.username}"
        except:
            return self.modified_by.get_full_name() or self.modified_by.username

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