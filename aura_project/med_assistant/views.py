from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.db.models import Q, Count, Avg
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
from datetime import datetime, timedelta
from django.utils import timezone
import calendar


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
    
    # Statistiques des prédictions du modèle
    prediction_stats = {}
    prediction_counts = Observation.objects.filter(model_prediction__isnull=False).values('model_prediction').annotate(count=Count('model_prediction'))
    for item in prediction_counts:
        pathology_name = Observation.get_pathology_display_name(item['model_prediction'])
        prediction_stats[f"Classe {item['model_prediction']} - {pathology_name}"] = item['count']
    
    # Statistiques des entités
    entity_stats = defaultdict(int)
    observations_with_entities = Observation.objects.exclude(entites={})
    
    for obs in observations_with_entities:
        if obs.entites:
            for entity_type, entities in obs.entites.items():
                entity_stats[entity_type] += len(entities) if isinstance(entities, list) else 1
    
    # Patients récents
    patients_recents = Patient.objects.order_by('-created_at')[:5]
    
    # Statistiques par mois (derniers 6 mois)
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
    
    # Nouvelles métriques avancées
    # Taux de traitement réussi
    total_processed = Observation.objects.filter(traitement_termine=True).count()
    success_rate = (total_processed / total_observations * 100) if total_observations > 0 else 0
    
    # Moyenne d'observations par patient
    avg_obs_per_patient = total_observations / total_patients if total_patients > 0 else 0
    
    # Patients avec observations récentes (derniers 30 jours)
    recent_threshold = timezone.now() - timedelta(days=30)
    active_patients = Patient.objects.filter(observations__date__gte=recent_threshold).distinct().count()
    
    context = {
        'total_patients': total_patients,
        'total_observations': total_observations,
        'observations_recentes': observations_recentes,
        'themes_stats': themes_stats,
        'prediction_stats': prediction_stats,
        'entity_stats': dict(entity_stats),
        'patients_recents': patients_recents,
        'monthly_stats': monthly_stats,
        'nlp_status': nlp_status,
        # Nouvelles métriques
        'success_rate': round(success_rate, 1),
        'avg_obs_per_patient': round(avg_obs_per_patient, 1),
        'active_patients': active_patients,
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
    """Détail d'un patient avec ses observations et résumé automatique"""
    patient = get_object_or_404(Patient, id=patient_id)
    observations = patient.observations.all()
    
    # Génération du résumé automatique du patient
    patient_summary = generate_patient_summary(patient)
    
    # Statistiques du patient
    patient_stats = get_patient_statistics(patient)
    
    # Timeline des observations
    timeline_data = get_patient_timeline(patient)
    
    # Pagination des observations
    paginator = Paginator(observations, 5)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'patient': patient,
        'observations': page_obj,
        'page_obj': page_obj,
        'patient_summary': patient_summary,
        'patient_stats': patient_stats,
        'timeline_data': timeline_data,
    }
    
    return render(request, 'med_assistant/patient_detail.html', context)


def generate_patient_summary(patient):
    """Génère un résumé automatique du patient basé sur son historique"""
    observations = patient.observations.filter(traitement_termine=True)
    
    if not observations.exists():
        return {
            'summary': "Aucune observation traitée disponible pour ce patient.",
            'total_observations': 0,
            'themes': [],
            'recent_activity': None,
            'risk_level': 'Inconnu'
        }
    
    # Analyse des thèmes principaux
    theme_counts = observations.values('theme_classe').annotate(count=Count('theme_classe')).order_by('-count')
    main_themes = []
    for theme_data in theme_counts[:3]:
        if theme_data['theme_classe']:
            theme_display = dict(Observation.THEME_CHOICES).get(theme_data['theme_classe'], theme_data['theme_classe'])
            main_themes.append({
                'name': theme_display,
                'count': theme_data['count'],
                'code': theme_data['theme_classe']
            })
    
    # Analyse des prédictions du modèle
    prediction_counts = observations.filter(model_prediction__isnull=False).values('model_prediction').annotate(count=Count('model_prediction'))
    prediction_analysis = []
    for pred_data in prediction_counts:
        pathology_name = Observation.get_pathology_display_name(pred_data['model_prediction'])
        prediction_analysis.append({
            'prediction': pred_data['model_prediction'],
            'pathology': pathology_name,
            'count': pred_data['count']
        })
    
    # Activité récente
    recent_obs = observations.order_by('-date').first()
    recent_activity = None
    if recent_obs:
        days_ago = (timezone.now() - recent_obs.date).days
        recent_activity = {
            'date': recent_obs.date,
            'days_ago': days_ago,
            'theme': recent_obs.get_theme_classe_display() if recent_obs.theme_classe else 'Non classé'
        }
    
    # Évaluation du niveau de risque basé sur la fréquence et les thèmes
    risk_level = calculate_risk_level(patient, observations)
    
    # Génération du résumé textuel
    summary_text = generate_summary_text(patient, main_themes, prediction_analysis, observations.count())
    
    return {
        'summary': summary_text,
        'total_observations': observations.count(),
        'themes': main_themes,
        'predictions': prediction_analysis,
        'recent_activity': recent_activity,
        'risk_level': risk_level,
        'first_visit': observations.order_by('date').first().date if observations.exists() else None,
        'last_visit': observations.order_by('-date').first().date if observations.exists() else None,
    }


def calculate_risk_level(patient, observations):
    """Calcule le niveau de risque du patient"""
    if not observations.exists():
        return 'Inconnu'
    
    # Facteurs de risque
    total_obs = observations.count()
    recent_obs = observations.filter(date__gte=timezone.now() - timedelta(days=30)).count()
    
    # Thèmes à risque élevé
    high_risk_themes = ['cardiovasculaire', 'diabete']
    high_risk_count = observations.filter(theme_classe__in=high_risk_themes).count()
    
    # Calcul du score de risque
    risk_score = 0
    
    # Fréquence des visites
    if total_obs > 10:
        risk_score += 2
    elif total_obs > 5:
        risk_score += 1
    
    # Activité récente
    if recent_obs > 3:
        risk_score += 2
    elif recent_obs > 1:
        risk_score += 1
    
    # Thèmes à risque
    if high_risk_count > total_obs * 0.5:
        risk_score += 2
    elif high_risk_count > 0:
        risk_score += 1
    
    # Âge du patient
    if patient.age > 70:
        risk_score += 1
    elif patient.age > 60:
        risk_score += 0.5
    
    # Classification du risque
    if risk_score >= 5:
        return 'Élevé'
    elif risk_score >= 3:
        return 'Modéré'
    elif risk_score >= 1:
        return 'Faible'
    else:
        return 'Très faible'


def generate_summary_text(patient, themes, predictions, total_obs):
    """Génère le texte de résumé du patient"""
    summary_parts = []
    
    # Introduction
    summary_parts.append(f"{patient.nom_complet}, {patient.age} ans, suivi avec {total_obs} observation{'s' if total_obs > 1 else ''}.")
    
    # Thèmes principaux
    if themes:
        main_theme = themes[0]
        if len(themes) == 1:
            summary_parts.append(f"Principalement suivi pour {main_theme['name'].lower()} ({main_theme['count']} observation{'s' if main_theme['count'] > 1 else ''}).")
        else:
            theme_list = ', '.join([t['name'].lower() for t in themes[:2]])
            summary_parts.append(f"Suivi principalement pour {theme_list}.")
    
    # Prédictions du modèle IA
    if predictions:
        main_prediction = predictions[0]
        summary_parts.append(f"L'IA identifie principalement des problématiques {main_prediction['pathology'].lower()}.")
    
    return ' '.join(summary_parts)


def get_patient_statistics(patient):
    """Calcule les statistiques détaillées du patient"""
    observations = patient.observations.all()
    
    if not observations.exists():
        return {}
    
    # Statistiques temporelles
    first_visit = observations.order_by('date').first().date
    last_visit = observations.order_by('-date').first().date
    follow_up_duration = (last_visit - first_visit).days
    
    # Fréquence des visites
    avg_days_between_visits = follow_up_duration / max(observations.count() - 1, 1)
    
    # Répartition par mois (derniers 12 mois)
    monthly_activity = []
    for i in range(12):
        month_start = timezone.now().replace(day=1) - timedelta(days=30*i)
        month_end = month_start.replace(day=1) + timedelta(days=32)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        
        count = observations.filter(date__gte=month_start, date__lte=month_end).count()
        monthly_activity.append({
            'month': calendar.month_name[month_start.month],
            'year': month_start.year,
            'count': count
        })
    
    monthly_activity.reverse()
    
    # Évolution des thèmes
    theme_evolution = observations.filter(theme_classe__isnull=False).values('theme_classe', 'date').order_by('date')
    
    return {
        'first_visit': first_visit,
        'last_visit': last_visit,
        'follow_up_duration_days': follow_up_duration,
        'avg_days_between_visits': round(avg_days_between_visits, 1),
        'monthly_activity': monthly_activity,
        'theme_evolution': list(theme_evolution),
        'total_processed': observations.filter(traitement_termine=True).count(),
        'success_rate': round(observations.filter(traitement_termine=True).count() / observations.count() * 100, 1) if observations.count() > 0 else 0
    }


def get_patient_timeline(patient):
    """Génère les données de timeline pour le patient"""
    observations = patient.observations.order_by('date')
    
    timeline_events = []
    for obs in observations:
        event = {
            'date': obs.date,
            'title': f"Observation - {obs.get_theme_classe_display() if obs.theme_classe else 'Non classé'}",
            'description': obs.resume[:100] + '...' if obs.resume and len(obs.resume) > 100 else obs.resume or 'Résumé non disponible',
            'theme': obs.theme_classe,
            'prediction': obs.model_prediction,
            'id': obs.id,
            'processed': obs.traitement_termine
        }
        timeline_events.append(event)
    
    return timeline_events


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
    """Création d'une nouvelle observation avec traitement NLP via FastAPI"""
    if request.method == 'POST':
        form = ObservationForm(request.POST, request.FILES)
        if form.is_valid():
            observation = form.save(commit=False)
            observation.save()
            
            # Traitement NLP avec FastAPI intégré
            try:
                results = nlp_pipeline.process_observation(observation)
                
                if results['success']:
                    observation.transcription = results['transcription']
                    observation.theme_classe = results['theme_classe']
                    observation.model_prediction = results['model_prediction']
                    observation.resume = results['resume']
                    observation.entites = results['entites']
                    observation.traitement_termine = True
                    
                    # Message de succès avec info sur la méthode utilisée et prédiction
                    if results.get('fastapi_used'):
                        pred_info = f" (Prédiction: {results['model_prediction']})" if results['model_prediction'] is not None else ""
                        messages.success(request, f'Observation créée et traitée avec succès via FastAPI{pred_info}.')
                    else:
                        messages.success(request, 'Observation créée et traitée avec succès (mode local).')
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
    """Détail d'une observation avec affichage de la prédiction"""
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
    
    context = {
        'observation': observation,
        'prediction_info': prediction_info
    }
    return render(request, 'med_assistant/observation_detail.html', context)


@require_http_methods(["POST"])
def observation_reprocess(request, observation_id):
    """Retraitement d'une observation avec FastAPI"""
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
        
        # Nouveau traitement avec FastAPI
        results = nlp_pipeline.process_observation(observation)
        
        if results['success']:
            observation.transcription = results['transcription']
            observation.theme_classe = results['theme_classe']
            observation.model_prediction = results['model_prediction']
            observation.resume = results['resume']
            observation.entites = results['entites']
            observation.traitement_termine = True
            
            # Message avec info sur la méthode utilisée et prédiction
            if results.get('fastapi_used'):
                pred_info = f" (Prédiction: {results['model_prediction']})" if results['model_prediction'] is not None else ""
                messages.success(request, f'Observation retraitée avec succès via FastAPI{pred_info}.')
            else:
                messages.success(request, 'Observation retraitée avec succès (mode local).')
        else:
            observation.traitement_erreur = results['error']
            messages.error(request, f'Erreur lors du retraitement: {results["error"]}')
        
        observation.save()
        
    except Exception as e:
        messages.error(request, f'Erreur lors du retraitement: {e}')
    
    return redirect('med_assistant:observation_detail', observation_id=observation.id)


def statistics(request):
    """Vue des statistiques avancées avec info FastAPI et prédictions"""
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
    
    # Statistiques des entités par catégorie
    entity_stats = defaultdict(lambda: defaultdict(int))
    observations_with_entities = Observation.objects.exclude(entites={})
    
    for obs in observations_with_entities:
        if obs.entites:
            for entity_type, entities in obs.entites.items():
                if isinstance(entities, list):
                    for entity in entities:
                        entity_stats[entity_type][entity] += 1
                else:
                    entity_stats[entity_type][entities] += 1
    
    # Top entités par catégorie
    top_entities = {}
    for category, entities in entity_stats.items():
        sorted_entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)[:5]
        top_entities[category] = sorted_entities
    
    # Nouvelles statistiques avancées
    
    # Évolution temporelle des prédictions (derniers 6 mois)
    prediction_evolution = []
    for i in range(6):
        month_start = timezone.now().replace(day=1) - timedelta(days=30*i)
        month_end = month_start.replace(day=1) + timedelta(days=32)
        month_end = month_end.replace(day=1) - timedelta(days=1)
        
        month_predictions = Observation.objects.filter(
            date__gte=month_start,
            date__lte=month_end,
            model_prediction__isnull=False
        ).values('model_prediction').annotate(count=Count('model_prediction'))
        
        month_data = {
            'month': calendar.month_name[month_start.month],
            'predictions': {item['model_prediction']: item['count'] for item in month_predictions}
        }
        prediction_evolution.append(month_data)
    
    prediction_evolution.reverse()
    
    # Analyse de performance du modèle
    total_observations = Observation.objects.count()
    processed_observations = Observation.objects.filter(traitement_termine=True).count()
    failed_observations = Observation.objects.filter(traitement_erreur__isnull=False).count()
    
    model_performance = {
        'total': total_observations,
        'processed': processed_observations,
        'failed': failed_observations,
        'success_rate': round(processed_observations / total_observations * 100, 1) if total_observations > 0 else 0,
        'failure_rate': round(failed_observations / total_observations * 100, 1) if total_observations > 0 else 0
    }
    
    # Analyse des patients à risque
    high_risk_patients = []
    for patient in Patient.objects.all():
        patient_summary = generate_patient_summary(patient)
        if patient_summary['risk_level'] in ['Élevé', 'Modéré']:
            high_risk_patients.append({
                'patient': patient,
                'risk_level': patient_summary['risk_level'],
                'total_observations': patient_summary['total_observations'],
                'main_themes': patient_summary['themes'][:2]
            })
    
    # Trier par niveau de risque
    risk_order = {'Élevé': 3, 'Modéré': 2, 'Faible': 1, 'Très faible': 0}
    high_risk_patients.sort(key=lambda x: risk_order.get(x['risk_level'], 0), reverse=True)
    
    # Statut de la pipeline NLP
    nlp_status = nlp_pipeline.get_status()
    
    context = {
        'theme_stats': theme_stats,
        'prediction_stats': prediction_stats,
        'entity_stats': dict(entity_stats),
        'top_entities': top_entities,
        'entity_categories': Observation.ENTITY_CATEGORIES,
        'nlp_status': nlp_status,
        # Nouvelles statistiques avancées
        'prediction_evolution': prediction_evolution,
        'model_performance': model_performance,
        'high_risk_patients': high_risk_patients[:10],  # Top 10
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


def patient_summary_api(request, patient_id):
    """API pour récupérer le résumé d'un patient"""
    try:
        patient = get_object_or_404(Patient, id=patient_id)
        summary = generate_patient_summary(patient)
        return JsonResponse({
            'success': True,
            'summary': summary
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        })