from django.urls import path
from . import views

app_name = 'med_assistant'

urlpatterns = [
    # Dashboard
    path('', views.dashboard, name='dashboard'),
    
    # Patients
    path('patients/', views.patient_list, name='patient_list'),
    path('patients/nouveau/', views.patient_create, name='patient_create'),
    path('patients/<int:patient_id>/', views.patient_detail, name='patient_detail'),
    path('patients/<int:patient_id>/modifier/', views.patient_edit, name='patient_edit'),
    
    # Observations
    path('observations/nouvelle/', views.observation_create, name='observation_create'),
    path('observations/<int:observation_id>/', views.observation_detail, name='observation_detail'),
    path('observations/<int:observation_id>/retraiter/', views.observation_reprocess, name='observation_reprocess'),
    
    # Statistiques
    path('statistiques/', views.statistics, name='statistics'),
    
    # API
    path('api/patients/search/', views.api_patient_search, name='api_patient_search'),
]