from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.db.models import Q, Count
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from .models import Patient, Observation
from .forms import PatientForm, ObservationForm, PatientSearchForm
from .nlp_pipeline import nlp_pipeline
from .api_client import fastapi_client
import json
from collections import defaultdict
from faster_whisper import WhisperModel
from django.views.decorators.csrf import csrf_exempt
import tempfile, subprocess, json



def dashboard(request):
    """Vue principale du dashboard AURA avec statistiques avancées"""
    # Statistiques générales
    total_patients = Patient.objects.count()
    total_observations = Observation.objects.count()
    observations_recentes = Observation.objects.select_related('patient')[:5]
    
    # Répartition par thème avec mapping des pathologies
    themes_stats = {}
    theme_counts = Observation.objects.filter(theme_classe__isnull=False).values('theme_classe').annotate(count=Count('theme_classe'))
    for item in theme_counts:
        theme_display = dict(Observation.THEME_CHOICES).get(item['theme_classe'], item['theme_classe'])
        themes_stats[theme_display] = item['count']
    
    # Statistiques des entités RÉELLES (seulement si on a des vraies entités)
    entity_stats = defaultdict(int)
    observations_with_entities = Observation.objects.exclude(entites={})
    
    # Compter seulement les vraies entités DrBERT
    real_entities_count = 0
    for obs in observations_with_entities:
        if obs.entites:
            for entity_type, entities in obs.entites.items():
                if isinstance(entities, list) and entities:  # Vérifier que la liste n'est pas vide
                    entity_stats[entity_type] += len(entities)
                    real_entities_count += len(entities)
                elif entities:  # Si c'est une string non vide
                    entity_stats[entity_type] += 1
                    real_entities_count += 1
    
    # Patients récents
    patients_recents = Patient.objects.order_by('-created_at')[:5]
    
    # Statistiques par mois (derniers 6 mois)
    from django.utils import timezone
    from datetime import timedelta
    import calendar
    
    six_months_ago = timezone.now() - timedelta(days=180)
    monthly_stats = []
    
    for i in range(6):
        month_start = timezone.now().replace(day=1) - timedelta(days=30*i)
        month_end = month_start.replace(day=1) + timedelta(days=32)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        
        count = Observation.objects.filter(
            date__gte=month_start,
            date__lte=month_end
        ).count()
        
        monthly_stats.append({
            'month': calendar.month_name[month_start.month],
            'count': count
        })
    
    monthly_stats.reverse()
    
    # Statut de la pipeline NLP
    nlp_status = nlp_pipeline.get_status()
    
    context = {
        'total_patients': total_patients,
        'total_observations': total_observations,
        'observations_recentes': observations_recentes,
        'themes_stats': themes_stats,
        'entity_stats': dict(entity_stats) if real_entities_count > 0 else {},  # Vide si pas de vraies entités
        'real_entities_count': real_entities_count,  # Nouveau : nombre total d'entités réelles
        'patients_recents': patients_recents,
        'monthly_stats': monthly_stats,
        'nlp_status': nlp_status,
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
    """Création d'une nouvelle observation avec traitement NLP via FastAPI, DrBERT et T5"""
    if request.method == 'POST':
        form = ObservationForm(request.POST, request.FILES)
        if form.is_valid():
            observation = form.save(commit=False)
            observation.save()
            
            # Traitement NLP avec FastAPI + DrBERT + T5 intégré
            try:
                results = nlp_pipeline.process_observation(observation)
                
                if results['success']:
                    observation.transcription = results['transcription']
                    observation.theme_classe = results['theme_classe']
                    observation.model_prediction = results['model_prediction']
                    observation.resume = results['resume']
                    observation.entites = results['entites']
                    observation.traitement_termine = True
                    
                    # Message de succès avec info sur les méthodes utilisées
                    methods_used = []
                    if results.get('fastapi_used'):
                        methods_used.append("FastAPI")
                    if results.get('drbert_used'):
                        methods_used.append("DrBERT")
                    if results.get('t5_used'):
                        methods_used.append("T5")
                    if not methods_used:
                        methods_used.append("Local")
                    
                    pred_info = f" (Prédiction: {results['model_prediction']})" if results['model_prediction'] is not None else ""
                    entities_info = f" - {sum(len(v) for v in results['entites'].values())} entités extraites" if results['entites'] else ""
                    
                    messages.success(request, f'Observation créée et traitée avec succès via {", ".join(methods_used)}{pred_info}{entities_info}.')
                else:
                    observation.traitement_erreur = results['error']
                    messages.warning(request, f'Observation créée mais erreur lors du traitement NLP: {results["error"]}')
                
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
    
    # Ajouter le statut de la pipeline au contexte
    nlp_status = nlp_pipeline.get_status()
    context = {
        'form': form,
        'nlp_status': nlp_status
    }
    return render(request, 'med_assistant/observation_form.html', context)


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


def observation_detail(request, observation_id):
    """Détail d'une observation avec affichage de la prédiction et entités DrBERT"""
    observation = get_object_or_404(
        Observation.objects.select_related('patient'), 
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
    
    # Compter les entités réelles
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


@require_http_methods(["POST"])
def observation_reprocess(request, observation_id):
    """Retraitement d'une observation avec FastAPI, DrBERT et T5"""
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
        
        # Nouveau traitement avec FastAPI + DrBERT + T5
        results = nlp_pipeline.process_observation(observation)
        
        if results['success']:
            observation.transcription = results['transcription']
            observation.theme_classe = results['theme_classe']
            observation.model_prediction = results['model_prediction']
            observation.resume = results['resume']
            observation.entites = results['entites']
            observation.traitement_termine = True
            
            # Message avec info sur les méthodes utilisées
            methods_used = []
            if results.get('fastapi_used'):
                methods_used.append("FastAPI")
            if results.get('drbert_used'):
                methods_used.append("DrBERT")
            if results.get('t5_used'):
                methods_used.append("T5")
            if not methods_used:
                methods_used.append("Local")
            
            pred_info = f" (Prédiction: {results['model_prediction']})" if results['model_prediction'] is not None else ""
            entities_info = f" - {sum(len(v) for v in results['entites'].values())} entités extraites" if results['entites'] else ""
            
            messages.success(request, f'Observation retraitée avec succès via {", ".join(methods_used)}{pred_info}{entities_info}.')
        else:
            observation.traitement_erreur = results['error']
            messages.error(request, f'Erreur lors du retraitement: {results["error"]}')
        
        observation.save()
        
    except Exception as e:
        messages.error(request, f'Erreur lors du retraitement: {e}')
    
    return redirect('med_assistant:observation_detail', observation_id=observation.id)


def statistics(request):
    """Vue des statistiques avancées avec info FastAPI, DrBERT, T5 et prédictions"""
    # Statistiques par thème avec mapping des pathologies
    theme_stats = {}
    theme_counts = Observation.objects.filter(theme_classe__isnull=False).values('theme_classe').annotate(count=Count('theme_classe'))
    for item in theme_counts:
        theme_display = dict(Observation.THEME_CHOICES).get(item['theme_classe'], item['theme_classe'])
        theme_stats[theme_display] = item['count']
    
    # Statistiques des prédictions du modèle
    prediction_stats = {}
    prediction_counts = Observation.objects.filter(model_prediction__isnull=False).values('model_prediction').annotate(count=Count('model_prediction'))
    for item in prediction_counts:
        pathology_name = Observation.get_pathology_display_name(item['model_prediction'])
        prediction_stats[f"Classe {item['model_prediction']} - {pathology_name}"] = item['count']
    
    # Statistiques des entités RÉELLES par catégorie DrBERT
    entity_stats = defaultdict(lambda: defaultdict(int))
    observations_with_entities = Observation.objects.exclude(entites={})
    
    total_real_entities = 0
    for obs in observations_with_entities:
        if obs.entites:
            for entity_type, entities in obs.entites.items():
                if isinstance(entities, list) and entities:
                    for entity in entities:
                        entity_stats[entity_type][entity] += 1
                        total_real_entities += 1
                elif entities:  # String non vide
                    entity_stats[entity_type][entities] += 1
                    total_real_entities += 1
    
    # Top entités par catégorie (seulement si on a des vraies entités)
    top_entities = {}
    if total_real_entities > 0:
        for category, entities in entity_stats.items():
            sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]
            top_entities[category] = sorted_entities
    
    # Statut de la pipeline NLP
    nlp_status = nlp_pipeline.get_status()
    
    # Catégories d'entités DrBERT
    drbert_categories = [
        ('DISO', 'Disorders - Maladies/Symptômes'),
        ('CHEM', 'Chemical/Drug - Médicaments'),
        ('ANAT', 'Anatomie - Parties du corps'),
        ('PROC', 'Procédure médicale'),
    ]
    
    context = {
        'theme_stats': theme_stats,
        'prediction_stats': prediction_stats,
        'entity_stats': dict(entity_stats) if total_real_entities > 0 else {},
        'top_entities': top_entities,
        'total_real_entities': total_real_entities,
        'entity_categories': drbert_categories,
        'nlp_status': nlp_status,
    }
    
    return render(request, 'med_assistant/statistics.html', context)


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


def api_nlp_status(request):
    """API pour récupérer le statut de la pipeline NLP"""
    status = nlp_pipeline.get_status()
    return JsonResponse(status)


def api_fastapi_models(request):
    """API pour récupérer les modèles disponibles via FastAPI"""
    try:
        models = fastapi_client.get_available_models()
        return JsonResponse({
            'success': True,
            'models': models,
            'api_available': fastapi_client.is_api_available()
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e),
            'api_available': False
        })