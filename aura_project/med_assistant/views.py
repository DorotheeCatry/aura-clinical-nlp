from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.models import User
from django.db.models import Q, Count
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from django.views.decorators.csrf import csrf_exempt
from django.urls import reverse_lazy
from .models import Patient, Observation, UserProfile
from .forms import PatientForm, ObservationForm, PatientSearchForm, ObservationEditForm, CustomLoginForm
from .nlp_pipeline import nlp_pipeline
import json
from collections import defaultdict
from faster_whisper import WhisperModel
import tempfile, subprocess, json
from datetime import date, datetime, timedelta
from django.utils import timezone
from med_assistant.utils.nlp_status import NLPStatus
import calendar


class UsernameBackend(ModelBackend):
    """Backend d'authentification par username (pas email)"""
    def authenticate(self, request, username=None, password=None, **kwargs):
        try:
            # Chercher par username uniquement
            user = User.objects.get(username=username)
            if user.check_password(password):
                return user
        except User.DoesNotExist:
            return None
        return None


class CustomLoginView(LoginView):
    """Vue de connexion personnalisée avec style AURA"""
    form_class = CustomLoginForm
    template_name = 'med_assistant/auth/login.html'
    redirect_authenticated_user = True
    
    def get_success_url(self):
        return reverse_lazy('med_assistant:dashboard')


def custom_logout(request):
    """Vue de déconnexion personnalisée sans message"""
    logout(request)
    return redirect('med_assistant:login')


@login_required
def dashboard(request):
    """Dashboard hospitalier avec métriques médicales"""
    # Statistiques générales
    total_patients = Patient.objects.count()
    total_observations = Observation.objects.count()
    
    # Patients par tranche d'âge
    today = date.today()
    age_groups = {
        'Enfants (0-17 ans)': 0,
        'Adultes (18-64 ans)': 0,
        'Seniors (65+ ans)': 0
    }
    
    for patient in Patient.objects.all():
        age = today.year - patient.date_naissance.year - ((today.month, today.day) < (patient.date_naissance.month, patient.date_naissance.day))
        if age < 18:
            age_groups['Enfants (0-17 ans)'] += 1
        elif age < 65:
            age_groups['Adultes (18-64 ans)'] += 1
        else:
            age_groups['Seniors (65+ ans)'] += 1
    
    # Activité des 7 derniers jours
    seven_days_ago = timezone.now() - timedelta(days=7)
    observations_semaine = Observation.objects.filter(date__gte=seven_days_ago).count()
    nouveaux_patients_semaine = Patient.objects.filter(created_at__gte=seven_days_ago).count()
    
    # Répartition par spécialité médicale
    specialites_stats = {}
    specialite_counts = Observation.objects.filter(theme_classe__isnull=False).values('theme_classe').annotate(count=Count('theme_classe'))
    for item in specialite_counts:
        specialite_display = dict(Observation.THEME_CHOICES).get(item['theme_classe'], item['theme_classe'])
        specialites_stats[specialite_display] = item['count']
    
    # Statistiques par jour de la semaine - TOUS LES JOURS FORCÉS
    weekday_stats = {
        'Lundi': 0, 'Mardi': 0, 'Mercredi': 0, 'Jeudi': 0, 
        'Vendredi': 0, 'Samedi': 0, 'Dimanche': 0
    }
    
    weekday_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    for obs in Observation.objects.all():
        weekday = weekday_names[obs.date.weekday()]
        weekday_stats[weekday] += 1
    
    # Top médicaments mentionnés (entités)
    medicaments_stats = defaultdict(int)
    observations_with_entities = Observation.objects.exclude(entites={})
    
    for obs in observations_with_entities:
        if obs.entites and 'Médicaments' in obs.entites:
            medicaments = obs.entites['Médicaments']
            if isinstance(medicaments, list):
                for med in medicaments:
                    medicaments_stats[med] += 1
    
    # Top 10 médicaments
    top_medicaments = dict(sorted(medicaments_stats.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Pathologies fréquentes (entités maladies)
    pathologies_stats = defaultdict(int)
    for obs in observations_with_entities:
        if obs.entites and 'Maladies et Symptômes' in obs.entites:
            pathologies = obs.entites['Maladies et Symptômes']
            if isinstance(pathologies, list):
                for path in pathologies:
                    pathologies_stats[path] += 1
    
    # Top 10 pathologies
    top_pathologies = dict(sorted(pathologies_stats.items(), key=lambda x: x[1], reverse=True)[:10])
    
    # Évolution mensuelle des consultations (6 derniers mois)
    monthly_consultations = []
    for i in range(6):
        month_start = timezone.now().replace(day=1) - timedelta(days=30*i)
        month_end = month_start.replace(day=1) + timedelta(days=32)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        
        count = Observation.objects.filter(
            date__gte=month_start,
            date__lte=month_end
        ).count()
        
        monthly_consultations.append({
            'month': calendar.month_name[month_start.month],
            'count': count
        })
    
    monthly_consultations.reverse()
    
    # Observations récentes
    observations_recentes = Observation.objects.select_related('patient', 'created_by')[:5]
    
    context = {
        'total_patients': total_patients,
        'total_observations': total_observations,
        'age_groups': age_groups,
        'observations_semaine': observations_semaine,
        'nouveaux_patients_semaine': nouveaux_patients_semaine,
        'specialites_stats': specialites_stats,
        'weekday_stats': weekday_stats,  # AJOUTÉ pour les graphiques
        'top_medicaments': top_medicaments,
        'top_pathologies': top_pathologies,
        'monthly_consultations': monthly_consultations,
        'observations_recentes': observations_recentes,
    }
    
    return render(request, 'med_assistant/dashboard.html', context)


@login_required
def patient_list(request):
    """Liste des patients avec recherche et filtres avancés"""
    form = PatientSearchForm(request.GET)
    patients = Patient.objects.all()
    
    if form.is_valid():
        # Recherche par nom/prénom
        if form.cleaned_data['search']:
            search_term = form.cleaned_data['search']
            patients = patients.filter(
                Q(nom__icontains=search_term) | 
                Q(prenom__icontains=search_term)
            )
        
        # Filtre par thème médical (utilise THEME_CHOICES)
        if form.cleaned_data['theme_classe']:
            theme_classe = form.cleaned_data['theme_classe']
            patients = patients.filter(
                observations__theme_classe=theme_classe
            ).distinct()
    
    # Précharger les observations pour optimiser les requêtes
    patients = patients.prefetch_related('observations')
    
    # Pagination
    paginator = Paginator(patients, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'form': form,
        'page_obj': page_obj,
        'patients': page_obj,
    }
    
    return render(request, 'med_assistant/patient_list.html', context)


@login_required
def patient_detail(request, patient_id):
    """Détail d'un patient avec ses observations"""
    patient = get_object_or_404(Patient, id=patient_id)
    observations = patient.observations.select_related('created_by', 'modified_by').all()
    
    # Pagination des observations
    paginator = Paginator(observations, 5)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'patient': patient,
        'observations': page_obj,
        'page_obj': page_obj,
    }
    
    return render(request, 'med_assistant/patient_detail.html', context)


@login_required
def patient_create(request):
    """Création d'un nouveau patient"""
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            patient = form.save()
            messages.success(request, f'Patient {patient.nom_complet} créé avec succès.')
            return redirect('med_assistant:patient_detail', patient_id=patient.id)
    else:
        form = PatientForm()
    
    context = {'form': form}
    return render(request, 'med_assistant/patient_form.html', context)


@login_required
def patient_edit(request, patient_id):
    """Modification d'un patient"""
    patient = get_object_or_404(Patient, id=patient_id)
    
    if request.method == 'POST':
        form = PatientForm(request.POST, instance=patient)
        if form.is_valid():
            patient = form.save()
            messages.success(request, f'Patient {patient.nom_complet} modifié avec succès.')
            return redirect('med_assistant:patient_detail', patient_id=patient.id)
    else:
        form = PatientForm(instance=patient)
    
    context = {
        'form': form,
        'patient': patient,
        'is_edit': True
    }
    return render(request, 'med_assistant/patient_form.html', context)


@login_required
def patient_delete(request, patient_id):
    """Suppression d'un patient"""
    patient = get_object_or_404(Patient, id=patient_id)
    
    if request.method == 'POST':
        patient_name = patient.nom_complet
        observations_count = patient.observations.count()
        patient.delete()
        messages.success(request, f'Patient {patient_name} et ses {observations_count} observation(s) supprimé(s) avec succès.')
        return redirect('med_assistant:patient_list')
    
    context = {
        'patient': patient,
        'observations_count': patient.observations.count()
    }
    return render(request, 'med_assistant/patient_delete_confirm.html', context)


@login_required
def observation_create(request):
    """Création d'une nouvelle observation avec traitement NLP et traçabilité"""
    if request.method == 'POST':
        form = ObservationForm(request.POST, request.FILES)
        if form.is_valid():
            observation = form.save(commit=False)
            # TRAÇABILITÉ : Enregistrer qui a créé l'observation
            observation.created_by = request.user
            observation.save()
            
            # Traitement NLP avec modèles Hugging Face directs
            try:
                results = nlp_pipeline.process_observation(observation)
                
                if results['success']:
                    observation.transcription = results['transcription']
                    observation.theme_classe = results['theme_classe']
                    observation.model_prediction = results['model_prediction']
                    observation.resume = results['resume']
                    observation.entites = results['entites']
                    observation.traitement_termine = True
                    
                    # Message de succès simplifié
                    entities_info = f" - {sum(len(v) for v in results['entites'].values())} éléments extraits" if results['entites'] else ""
                    messages.success(request, f'Observation créée et analysée avec succès{entities_info}.')
                else:
                    observation.traitement_erreur = results['error']
                    messages.warning(request, f'Observation créée mais erreur lors de l\'analyse: {results["error"]}')
                
                observation.save()
                
            except Exception as e:
                messages.error(request, f'Erreur lors du traitement: {e}')
            
            return redirect('med_assistant:observation_detail', observation_id=observation.id)
    else:
        form = ObservationForm()
        # Pré-sélectionner un patient si fourni en paramètre
        patient_id = request.GET.get('patient_id')
        if patient_id:
            try:
                patient = Patient.objects.get(id=patient_id)
                form.initial['patient'] = patient
            except Patient.DoesNotExist:
                pass
    
    # Ajouter le statut de la pipeline au contexte avec NLPStatus
    raw_status = nlp_pipeline.get_status()
    nlp_status = NLPStatus(
        classification=raw_status.get("classification_available"),
        drbert=raw_status.get("drbert_available"),
        t5=raw_status.get("t5_available"),
        whisper=raw_status.get("whisper_available"),
    )
    
    context = {
        'form': form,
        'nlp_status': nlp_status
    }
    return render(request, 'med_assistant/observation_form.html', context)


@login_required
def observation_edit(request, observation_id):
    """Modification d'une observation avec traçabilité"""
    observation = get_object_or_404(Observation, id=observation_id)
    
    if request.method == 'POST':
        form = ObservationEditForm(request.POST, instance=observation)
        if form.is_valid():
            observation = form.save(commit=False)
            # TRAÇABILITÉ : Enregistrer qui a modifié l'observation
            observation.modified_by = request.user
            observation.save()
            messages.success(request, 'Observation modifiée avec succès.')
            return redirect('med_assistant:observation_detail', observation_id=observation.id)
    else:
        form = ObservationEditForm(instance=observation)
    
    context = {
        'form': form,
        'observation': observation,
        'is_edit': True
    }
    return render(request, 'med_assistant/observation_edit.html', context)


@login_required
def observation_delete(request, observation_id):
    """Suppression d'une observation"""
    observation = get_object_or_404(Observation, id=observation_id)
    
    if request.method == 'POST':
        patient = observation.patient
        observation_date = observation.date.strftime('%d/%m/%Y à %H:%M')
        observation.delete()
        messages.success(request, f'Observation du {observation_date} supprimée avec succès.')
        return redirect('med_assistant:patient_detail', patient_id=patient.id)
    
    context = {
        'observation': observation
    }
    return render(request, 'med_assistant/observation_delete_confirm.html', context)


@login_required
@require_http_methods(["POST"])
def delete_entity(request, observation_id):
    """Suppression d'une entité spécifique d'une observation"""
    observation = get_object_or_404(Observation, id=observation_id)
    
    try:
        # Récupérer les paramètres
        category = request.POST.get('category')
        entity_text = request.POST.get('entity')
        
        if not category or not entity_text:
            messages.error(request, 'Paramètres manquants pour la suppression.')
            return redirect('med_assistant:observation_detail', observation_id=observation.id)
        
        # Vérifier que l'observation a des entités
        if not observation.entites:
            messages.error(request, 'Aucune entité trouvée dans cette observation.')
            return redirect('med_assistant:observation_detail', observation_id=observation.id)
        
        # Vérifier que la catégorie existe
        if category not in observation.entites:
            messages.error(request, f'Catégorie "{category}" non trouvée.')
            return redirect('med_assistant:observation_detail', observation_id=observation.id)
        
        # Supprimer l'entité de la liste
        entities_list = observation.entites[category]
        if isinstance(entities_list, list):
            if entity_text in entities_list:
                entities_list.remove(entity_text)
                
                # Si la liste devient vide, supprimer la catégorie
                if not entities_list:
                    del observation.entites[category]
                
                # TRAÇABILITÉ : Enregistrer qui a modifié
                observation.modified_by = request.user
                observation.save()
                
                messages.success(request, f'Entité "{entity_text}" supprimée de la catégorie "{category}".')
            else:
                messages.error(request, f'Entité "{entity_text}" non trouvée dans la catégorie "{category}".')
        else:
            messages.error(request, 'Format d\'entités invalide.')
        
    except Exception as e:
        messages.error(request, f'Erreur lors de la suppression de l\'entité: {e}')
    
    return redirect('med_assistant:observation_detail', observation_id=observation.id)


@login_required
@csrf_exempt
def transcribe_audio(request):
    """Endpoint pour la transcription audio via Whisper local"""
    if request.method != "POST" or "audio" not in request.FILES:
        return JsonResponse({"error": "no_audio"}, status=400)

    uploaded = request.FILES["audio"]

    try:
        # Utiliser la pipeline NLP pour la transcription
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
            for chunk in uploaded.chunks():
                temp_file.write(chunk)
            temp_file.flush()
            
            # Utiliser la méthode de transcription de la pipeline
            transcription = nlp_pipeline.transcribe_audio(temp_file.name)
            
            # Nettoyer le fichier temporaire
            import os
            os.unlink(temp_file.name)
            
            if transcription:
                return JsonResponse({"text": transcription})
            else:
                return JsonResponse({"error": "transcription_failed"}, status=500)
                
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@login_required
def observation_detail(request, observation_id):
    """Détail d'une observation avec affichage de la prédiction et entités DrBERT"""
    observation = get_object_or_404(
        Observation.objects.select_related('patient', 'created_by', 'modified_by'), 
        id=observation_id
    )
    
    # Ajouter des informations sur la prédiction du modèle
    prediction_info = None
    if observation.model_prediction is not None:
        prediction_info = {
            'prediction': observation.model_prediction,
            'pathology_name': Observation.get_pathology_display_name(observation.model_prediction),
            'confidence': 'Élevée' if observation.model_prediction in [0, 1, 2] else 'Faible'
        }
    
    # Compter les entités réelles avec noms compréhensibles
    entities_count = 0
    if observation.entites:
        for entity_type, entities in observation.entites.items():
            if isinstance(entities, list):
                entities_count += len(entities)
            elif entities:
                entities_count += 1
    
    context = {
        'observation': observation,
        'prediction_info': prediction_info,
        'entities_count': entities_count
    }
    return render(request, 'med_assistant/observation_detail.html', context)


@login_required
@require_http_methods(["POST"])
def observation_reprocess(request, observation_id):
    """Retraitement d'une observation avec modèles Hugging Face directs"""
    observation = get_object_or_404(Observation, id=observation_id)
    
    try:
        # Réinitialiser les résultats précédents
        observation.transcription = None
        observation.theme_classe = None
        observation.model_prediction = None
        observation.resume = None
        observation.entites = {}
        observation.traitement_termine = False
        observation.traitement_erreur = None
        
        # TRAÇABILITÉ : Enregistrer qui a relancé le traitement
        observation.modified_by = request.user
        
        # Nouveau traitement avec modèles Hugging Face directs
        results = nlp_pipeline.process_observation(observation)
        
        if results['success']:
            observation.transcription = results['transcription']
            observation.theme_classe = results['theme_classe']
            observation.model_prediction = results['model_prediction']
            observation.resume = results['resume']
            observation.entites = results['entites']
            observation.traitement_termine = True
            
            # Message simplifié
            entities_info = f" - {sum(len(v) for v in results['entites'].values())} éléments extraits" if results['entites'] else ""
            messages.success(request, f'Observation retraitée avec succès{entities_info}.')
        else:
            observation.traitement_erreur = results['error']
            messages.error(request, f'Erreur lors du retraitement: {results["error"]}')
        
        observation.save()
        
    except Exception as e:
        messages.error(request, f'Erreur lors du retraitement: {e}')
    
    return redirect('med_assistant:observation_detail', observation_id=observation.id)


@login_required
def statistics(request):
    """Statistiques hospitalières détaillées"""
    # Statistiques générales
    total_patients = Patient.objects.count()
    total_observations = Observation.objects.count()
    
    # Répartition par âge
    today = date.today()
    age_groups = {
        'Enfants (0-17 ans)': 0,
        'Adultes (18-64 ans)': 0,
        'Seniors (65+ ans)': 0
    }
    
    for patient in Patient.objects.all():
        age = today.year - patient.date_naissance.year - ((patient.date_naissance.day) > (today.month, today.day))
        if age < 18:
            age_groups['Enfants (0-17 ans)'] += 1
        elif age < 65:
            age_groups['Adultes (18-64 ans)'] += 1
        else:
            age_groups['Seniors (65+ ans)'] += 1
    
    # Statistiques par spécialité médicale
    specialites_stats = {}
    specialite_counts = Observation.objects.filter(theme_classe__isnull=False).values('theme_classe').annotate(count=Count('theme_classe'))
    for item in specialite_counts:
        specialite_display = dict(Observation.THEME_CHOICES).get(item['theme_classe'], item['theme_classe'])
        specialites_stats[specialite_display] = item['count']
    
    # Analyse des médicaments
    medicaments_stats = defaultdict(int)
    observations_with_entities = Observation.objects.exclude(entites={})
    
    for obs in observations_with_entities:
        if obs.entites and 'Médicaments' in obs.entites:
            medicaments = obs.entites['Médicaments']
            if isinstance(medicaments, list):
                for med in medicaments:
                    medicaments_stats[med] += 1
    
    # Top 15 médicaments
    top_medicaments = dict(sorted(medicaments_stats.items(), key=lambda x: x[1], reverse=True)[:15])
    
    # Analyse des pathologies
    pathologies_stats = defaultdict(int)
    for obs in observations_with_entities:
        if obs.entites and 'Maladies et Symptômes' in obs.entites:
            pathologies = obs.entites['Maladies et Symptômes']
            if isinstance(pathologies, list):
                for path in pathologies:
                    pathologies_stats[path] += 1
    
    # Top 15 pathologies
    top_pathologies = dict(sorted(pathologies_stats.items(), key=lambda x: x[1], reverse=True)[:15])
    
    # Analyse des gestes/procédures médicales
    procedures_stats = defaultdict(int)
    for obs in observations_with_entities:
        if obs.entites and 'Procédures Médicales' in obs.entites:
            procedures = obs.entites['Procédures Médicales']
            if isinstance(procedures, list):
                for proc in procedures:
                    procedures_stats[proc] += 1
    
    # Top 15 procédures
    top_procedures = dict(sorted(procedures_stats.items(), key=lambda x: x[1], reverse=True)[:15])
    
    # Évolution mensuelle des consultations (12 derniers mois)
    monthly_evolution = []
    for i in range(12):
        month_start = timezone.now().replace(day=1) - timedelta(days=30*i)
        month_end = month_start.replace(day=1) + timedelta(days=32)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        
        observations_count = Observation.objects.filter(
            date__gte=month_start,
            date__lte=month_end
        ).count()
        
        patients_count = Patient.objects.filter(
            created_at__gte=month_start,
            created_at__lte=month_end
        ).count()
        
        monthly_evolution.append({
            'month': f"{calendar.month_name[month_start.month]} {month_start.year}",
            'observations': observations_count,
            'nouveaux_patients': patients_count
        })
    
    monthly_evolution.reverse()
    
    # Activité par jour de la semaine
    weekday_stats = {
        'Lundi': 0, 'Mardi': 0, 'Mercredi': 0, 'Jeudi': 0, 
        'Vendredi': 0, 'Samedi': 0, 'Dimanche': 0
    }
    
    weekday_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    for obs in Observation.objects.all():
        weekday = weekday_names[obs.date.weekday()]
        weekday_stats[weekday] += 1
    
    # Statistiques par praticien
    practitioner_stats = {}
    practitioner_counts = Observation.objects.filter(created_by__isnull=False).values('created_by__first_name', 'created_by__last_name').annotate(count=Count('created_by'))
    for item in practitioner_counts:
        name = f"{item['created_by__first_name']} {item['created_by__last_name']}"
        practitioner_stats[name] = item['count']
    
    context = {
        'total_patients': total_patients,
        'total_observations': total_observations,
        'age_groups': age_groups,
        'specialites_stats': specialites_stats,
        'top_medicaments': top_medicaments,
        'top_pathologies': top_pathologies,
        'top_procedures': top_procedures,
        'monthly_evolution': monthly_evolution,
        'weekday_stats': weekday_stats,
        'practitioner_stats': practitioner_stats,
        'total_medicaments': len(medicaments_stats),
        'total_pathologies': len(pathologies_stats),
        'total_procedures': len(procedures_stats),
    }
    
    return render(request, 'med_assistant/statistics.html', context)


@login_required
def api_patient_search(request):
    """API pour la recherche de patients (pour l'autocomplétion)"""
    query = request.GET.get('q', '')
    
    if len(query) < 2:
        return JsonResponse({'patients': []})
    
    patients = Patient.objects.filter(
        Q(nom__icontains=query) | Q(prenom__icontains=query)
    )[:10]
    
    patients_data = [
        {
            'id': p.id,
            'nom_complet': p.nom_complet,
            'date_naissance': p.date_naissance.strftime('%d/%m/%Y')
        }
        for p in patients
    ]
    
    return JsonResponse({'patients': patients_data})


@login_required
def api_nlp_status(request):
    """API pour récupérer le statut de la pipeline NLP"""
    status = nlp_pipeline.get_status()
    return JsonResponse(status)