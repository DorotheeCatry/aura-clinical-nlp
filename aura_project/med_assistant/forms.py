from django import forms
from .models import Patient, Observation


class PatientForm(forms.ModelForm):
    """Formulaire pour créer/modifier un patient"""
    
    class Meta:
        model = Patient
        fields = ['nom', 'prenom', 'date_naissance']
        widgets = {
            'nom': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors',
                'placeholder': 'Nom du patient'
            }),
            'prenom': forms.TextInput(attrs={
                'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors',
                'placeholder': 'Prénom du patient'
            }),
            'date_naissance': forms.DateInput(attrs={
                'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors',
                'type': 'date'
            }),
        }


class ObservationForm(forms.ModelForm):
    """Formulaire pour créer une nouvelle observation"""
    
    class Meta:
        model = Observation
        fields = ['patient', 'texte_saisi', 'audio_file']
        widgets = {
            'patient': forms.Select(attrs={
                'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors'
            }),
            'texte_saisi': forms.Textarea(attrs={
                'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors resize-none',
                'rows': 6,
                'placeholder': 'Saisissez votre note médicale ici...'
            }),
            'audio_file': forms.FileInput(attrs={
                'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100',
                'accept': 'audio/*'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['patient'].empty_label = "Sélectionner un patient"
        
    def clean(self):
        cleaned_data = super().clean()
        texte_saisi = cleaned_data.get('texte_saisi')
        audio_file = cleaned_data.get('audio_file')
        
        if not texte_saisi and not audio_file:
            raise forms.ValidationError(
                "Vous devez fournir soit un texte, soit un fichier audio."
            )
        
        return cleaned_data


class ObservationEditForm(forms.ModelForm):
    """Formulaire pour modifier une observation existante"""
    
    class Meta:
        model = Observation
        fields = ['theme_classe', 'resume', 'texte_saisi']
        widgets = {
            'theme_classe': forms.Select(attrs={
                'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors'
            }),
            'resume': forms.Textarea(attrs={
                'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors resize-none',
                'rows': 4,
                'placeholder': 'Résumé de l\'observation...'
            }),
            'texte_saisi': forms.Textarea(attrs={
                'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors resize-none',
                'rows': 8,
                'placeholder': 'Texte de l\'observation...'
            }),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['theme_classe'].empty_label = "Sélectionner un thème"


class PatientSearchForm(forms.Form):
    """Formulaire de recherche de patients"""
    
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors',
            'placeholder': 'Rechercher un patient (nom, prénom)...'
        })
    )