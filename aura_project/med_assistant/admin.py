from django.contrib import admin
from .models import Patient, Observation


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin):
    list_display = ('nom', 'prenom', 'date_naissance', 'created_at')
    list_filter = ('date_naissance', 'created_at')
    search_fields = ('nom', 'prenom')
    ordering = ('nom', 'prenom')


@admin.register(Observation)
class ObservationAdmin(admin.ModelAdmin):
    list_display = ('patient', 'date', 'theme_classe', 'traitement_termine')
    list_filter = ('theme_classe', 'traitement_termine', 'date')
    search_fields = ('patient__nom', 'patient__prenom', 'texte_saisi', 'transcription')
    readonly_fields = ('date', 'created_at', 'updated_at')
    
    fieldsets = (
        ('Patient et Date', {
            'fields': ('patient', 'date')
        }),
        ('Données d\'entrée', {
            'fields': ('texte_saisi', 'audio_file')
        }),
        ('Résultats NLP', {
            'fields': ('transcription', 'theme_classe', 'resume', 'entites'),
            'classes': ('collapse',)
        }),
        ('Traitement', {
            'fields': ('traitement_termine', 'traitement_erreur'),
            'classes': ('collapse',)
        }),
        ('Métadonnées', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )