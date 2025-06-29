from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from .models import Patient, Observation, UserProfile


class CustomLoginForm(AuthenticationForm):
    """Formulaire de connexion moderne style image"""
    
    username = forms.EmailField(
        label="Email address",
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'Enter your email'
        })
    )
    
    password = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'Enter your password'
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Permettre la connexion par email
        self.fields['username'].label = 'Email address'


class CustomUserCreationForm(UserCreationForm):
    """Formulaire d'inscription avec profil professionnel"""
    
    email = forms.EmailField(
        required=True,
        label="Email address",
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'Enter your email'
        })
    )
    
    first_name = forms.CharField(
        max_length=30,
        required=True,
        label="First name",
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'Enter your first name'
        })
    )
    
    last_name = forms.CharField(
        max_length=30,
        required=True,
        label="Last name",
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'Enter your last name'
        })
    )
    
    role = forms.ChoiceField(
        choices=UserProfile.ROLE_CHOICES,
        required=True,
        label="Professional role",
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all'
        })
    )
    
    specialite = forms.CharField(
        max_length=100,
        required=False,
        label="Specialty (optional)",
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'e.g., Cardiology, Emergency Medicine...'
        })
    )
    
    etablissement = forms.CharField(
        max_length=200,
        required=False,
        label="Institution (optional)",
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'Hospital, clinic, medical center...'
        })
    )

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Personnaliser les champs existants
        self.fields['username'].widget.attrs.update({
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'Choose a username'
        })
        
        self.fields['password1'].widget.attrs.update({
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'Create a password'
        })
        
        self.fields['password2'].widget.attrs.update({
            'class': 'w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all',
            'placeholder': 'Confirm your password'
        })
        
        # Labels en anglais pour correspondre au design
        self.fields['username'].label = 'Username'
        self.fields['password1'].label = 'Password'
        self.fields['password2'].label = 'Confirm password'

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        if commit:
            user.save()
            # Créer le profil utilisateur
            UserProfile.objects.create(
                user=user,
                role=self.cleaned_data['role'],
                specialite=self.cleaned_data.get('specialite', ''),
                etablissement=self.cleaned_data.get('etablissement', '')
            )
        return user


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
    """Formulaire de recherche et filtrage de patients"""
    
    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors',
            'placeholder': 'Rechercher un patient (nom, prénom)...'
        })
    )
    
    theme_classe = forms.ChoiceField(
        required=False,
        widget=forms.Select(attrs={
            'class': 'w-full px-4 py-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-colors'
        })
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Utiliser directement les THEME_CHOICES du modèle
        theme_choices = [('', 'Toutes les spécialités')] + list(Observation.THEME_CHOICES)
        self.fields['theme_classe'].choices = theme_choices