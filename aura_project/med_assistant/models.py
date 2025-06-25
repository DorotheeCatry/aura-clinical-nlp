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


class Observation(models.Model):
    """Modèle représentant une observation médicale avec traitement NLP"""
    
    THEME_CHOICES = [
        ('cardio', 'Cardiologie'),
        ('psy', 'Psychiatrie'),
        ('diabete', 'Diabète'),
        ('neuro', 'Neurologie'),
        ('pneumo', 'Pneumologie'),
        ('gastro', 'Gastroentérologie'),
        ('ortho', 'Orthopédie'),
        ('dermato', 'Dermatologie'),
        ('general', 'Médecine générale'),
        ('autre', 'Autre'),
    ]

    # Nouvelles catégories d'entités médicales
    ENTITY_CATEGORIES = [
        ('DISO', 'Disorder - Maladies/Symptômes'),
        ('CHEM', 'Chemical/Drug - Médicaments'),
        ('ANAT', 'Anatomie - Parties du corps'),
        ('PROC', 'Procédure médicale'),
        ('TEST', 'Examen médical'),
        ('MED', 'Médicament'),
        ('BODY', 'Partie du corps'),
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

    def get_theme_display_color(self):
        """Retourne la couleur associée au thème"""
        colors = {
            'cardio': 'bg-red-50 text-red-700 border-red-200',
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
        """Retourne la couleur pour un type d'entité"""
        colors = {
            'DISO': 'bg-red-50 text-red-700 border-red-200',
            'CHEM': 'bg-blue-50 text-blue-700 border-blue-200',
            'ANAT': 'bg-green-50 text-green-700 border-green-200',
            'PROC': 'bg-purple-50 text-purple-700 border-purple-200',
            'TEST': 'bg-orange-50 text-orange-700 border-orange-200',
            'MED': 'bg-cyan-50 text-cyan-700 border-cyan-200',
            'BODY': 'bg-emerald-50 text-emerald-700 border-emerald-200',
        }
        return colors.get(entity_type, 'bg-gray-50 text-gray-700 border-gray-200')