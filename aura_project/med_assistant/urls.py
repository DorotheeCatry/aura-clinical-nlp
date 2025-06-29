from django.urls import path
from . import views

app_name = 'med_assistant'

urlpatterns = [
    # Authentification (SANS register - app interne hôpital)
    path('login/', views.CustomLoginView.as_view(), name='login'),
    path('logout/', views.custom_logout, name='logout'),  # CORRIGÉ : fonction au lieu de classe
    
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Patients
    path('patients/', views.patient_list, name='patient_list'),
    path('patients/nouveau/', views.patient_create, name='patient_create'),
    path('patients/<int:patient_id>/', views.patient_detail, name='patient_detail'),
    path('patients/<int:patient_id>/modifier/', views.patient_edit, name='patient_edit'),
    path('patients/<int:patient_id>/supprimer/', views.patient_delete, name='patient_delete'),
    
    # Observations
    path('observations/nouvelle/', views.observation_create, name='observation_create'),
    path("api/transcribe/", views.transcribe_audio, name="transcribe"),
    path('observations/<int:observation_id>/', views.observation_detail, name='observation_detail'),
    path('observations/<int:observation_id>/modifier/', views.observation_edit, name='observation_edit'),
    path('observations/<int:observation_id>/supprimer/', views.observation_delete, name='observation_delete'),
    path('observations/<int:observation_id>/retraiter/', views.observation_reprocess, name='observation_reprocess'),
    
    # Entités
    path('observations/<int:observation_id>/entites/supprimer/', views.delete_entity, name='delete_entity'),
    
    # Statistiques
    path('statistiques/', views.statistics, name='statistics'),
    
    # API
    path('api/patients/search/', views.api_patient_search, name='api_patient_search'),
    path('api/nlp/status/', views.api_nlp_status, name='api_nlp_status'),
]