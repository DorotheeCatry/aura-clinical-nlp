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
        validators=[FileExtensionValidator(allowed_extensions=['mp3', 'wav', 'ogg', 'm4a'])],
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
            'cardio': 'bg-red-100 text-red-800',
            'psy': 'bg-purple-100 text-purple-800',
            'diabete': 'bg-yellow-100 text-yellow-800',
            'neuro': 'bg-indigo-100 text-indigo-800',
            'pneumo': 'bg-blue-100 text-blue-800',
            'gastro': 'bg-green-100 text-green-800',
            'ortho': 'bg-gray-100 text-gray-800',
            'dermato': 'bg-pink-100 text-pink-800',
            'general': 'bg-slate-100 text-slate-800',
            'autre': 'bg-neutral-100 text-neutral-800',
        }
        return colors.get(self.theme_classe, 'bg-gray-100 text-gray-800')