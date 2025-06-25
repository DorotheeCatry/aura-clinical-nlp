from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from .models import Patient, Observation
from .forms import PatientForm, ObservationForm, PatientSearchForm
from .nlp_pipeline import nlp_pipeline
import json


def dashboard(request):
    """Vue principale du dashboard AURA"""
    # Statistiques générales
    total_patients = Patient.objects.count()
    total_observations = Observation.objects.count()
    observations_recentes = Observation.objects.select_related('patient')[:5]
    
    # Répartition par thème
    themes_stats = {}
    for choice in Observation.THEME_CHOICES:
        count = Observation.objects.filter(theme_classe=choice[0]).count()
        if count > 0:
            themes_stats[choice[1]] = count
    
    context = {
        'total_patients': total_patients,
        'total_observations': total_observations,
        'observations_recentes': observations_recentes,
        'themes_stats': themes_stats,
    }
    
    return render(request, 'med_assistant/dashboard.html', context)


def patient_list(request):
    """Liste des patients avec recherche"""
    form = PatientSearchForm(request.GET)
    patients = Patient.objects.all()
    
    if form.is_valid() and form.cleaned_data['search']:
        search_term = form.cleaned_data['search']
        patients = patients.filter(
            Q(nom__icontains=search_term) | 
            Q(prenom__icontains=search_term)
        )
    
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


def patient_detail(request, patient_id):
    """Détail d'un patient avec ses observations"""
    patient = get_object_or_404(Patient, id=patient_id)
    observations = patient.observations.all()
    
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


def observation_create(request):
    """Création d'une nouvelle observation"""
    if request.method == 'POST':
        form = ObservationForm(request.POST, request.FILES)
        if form.is_valid():
            observation = form.save(commit=False)
            observation.save()
            
            # Traitement NLP immédiat (en production, utiliser une tâche asynchrone)
            try:
                results = nlp_pipeline.process_observation(observation)
                
                if results['success']:
                    observation.transcription = results['transcription']
                    observation.theme_classe = results['theme_classe']
                    observation.resume = results['resume']
                    observation.entites = results['entites']
                    observation.traitement_termine = True
                    messages.success(request, 'Observation créée et traitée avec succès.')
                else:
                    observation.traitement_erreur = results['error']
                    messages.warning(request, 'Observation créée mais erreur lors du traitement NLP.')
                
                observation.save()
                
            except Exception as e:
                messages.error(request, f'Erreur lors du traitement: {e}')
            
            return redirect('med_assistant:patient_detail', patient_id=observation.patient.id)
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
    
    context = {'form': form}
    return render(request, 'med_assistant/observation_form.html', context)


def observation_detail(request, observation_id):
    """Détail d'une observation"""
    observation = get_object_or_404(
        Observation.objects.select_related('patient'), 
        id=observation_id
    )
    
    context = {'observation': observation}
    return render(request, 'med_assistant/observation_detail.html', context)


@require_http_methods(["POST"])
def observation_reprocess(request, observation_id):
    """Retraitement d'une observation"""
    observation = get_object_or_404(Observation, id=observation_id)
    
    try:
        # Réinitialiser les résultats précédents
        observation.transcription = None
        observation.theme_classe = None
        observation.resume = None
        observation.entites = {}
        observation.traitement_termine = False
        observation.traitement_erreur = None
        
        # Nouveau traitement
        results = nlp_pipeline.process_observation(observation)
        
        if results['success']:
            observation.transcription = results['transcription']
            observation.theme_classe = results['theme_classe']
            observation.resume = results['resume']
            observation.entites = results['entites']
            observation.traitement_termine = True
            messages.success(request, 'Observation retraitée avec succès.')
        else:
            observation.traitement_erreur = results['error']
            messages.error(request, f'Erreur lors du retraitement: {results["error"]}')
        
        observation.save()
        
    except Exception as e:
        messages.error(request, f'Erreur lors du retraitement: {e}')
    
    return redirect('med_assistant:observation_detail', observation_id=observation.id)


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