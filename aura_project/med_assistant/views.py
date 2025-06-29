@login_required
def dashboard(request):
    """Dashboard hospitalier simplifié avec métriques essentielles et filtres"""
    # Filtre par spécialité
    selected_specialty = request.GET.get('specialty', '')
    
    # Statistiques générales
    total_patients = Patient.objects.count()
    total_observations = Observation.objects.count()
    
    # Consultations par jour de la semaine (7 derniers jours)
    today = timezone.now().date()
    weekly_consultations = {}
    weekday_names = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    
    # Calculer les consultations pour chaque jour
    max_consultations = 10  # VALEUR FIXE pour l'axe Y
    for i in range(7):
        day = today - timedelta(days=6-i)  # Commencer par lundi d'il y a 6 jours
        day_name = weekday_names[day.weekday()]
        
        # Appliquer le filtre de spécialité si sélectionné
        observations_query = Observation.objects.filter(date__date=day)
        if selected_specialty:
            observations_query = observations_query.filter(theme_classe=selected_specialty)
        
        count = observations_query.count()
        weekly_consultations[day_name] = count
    
    # Patients par tranche d'âge avec pourcentages
    today_date = date.today()
    age_groups = {
        'Enfants (0-17 ans)': 0,
        'Adultes (18-64 ans)': 0,
        'Seniors (65+ ans)': 0
    }
    
    # Filtrer les patients selon la spécialité si sélectionnée
    patients_query = Patient.objects.all()
    if selected_specialty:
        patients_query = patients_query.filter(observations__theme_classe=selected_specialty).distinct()
    
    for patient in patients_query:
        age = today_date.year - patient.date_naissance.year - ((today_date.month, today_date.day) < (patient.date_naissance.month, patient.date_naissance.day))
        if age < 18:
            age_groups['Enfants (0-17 ans)'] += 1
        elif age < 65:
            age_groups['Adultes (18-64 ans)'] += 1
        else:
            age_groups['Seniors (65+ ans)'] += 1
    
    # Calculer les pourcentages
    total_filtered_patients = sum(age_groups.values())
    age_groups_with_percent = {}
    for group, count in age_groups.items():
        percentage = (count / total_filtered_patients * 100) if total_filtered_patients > 0 else 0
        age_groups_with_percent[group] = {
            'count': count,
            'percentage': round(percentage, 1)
        }
    
    # Patients par service hospitalier (TOUTES les spécialités)
    service_patients = {}
    
    # Récupérer toutes les spécialités possibles depuis THEME_CHOICES
    all_specialties = dict(Observation.THEME_CHOICES)
    
    for theme_code, theme_display in all_specialties.items():
        # Compter les patients uniques pour chaque spécialité
        patient_count = Patient.objects.filter(
            observations__theme_classe=theme_code
        ).distinct().count()
        
        # Simplifier les noms des services et TOUJOURS afficher
        service_name = theme_display.replace('Psychique/Neuropsychiatrique', 'Psychiatrie').replace('Métabolique/Diabète', 'Endocrinologie')
        service_patients[service_name] = patient_count
    
    # Calculer le maximum pour normaliser les barres (minimum 10 pour l'échelle)
    max_patients_service = max(max(service_patients.values()) if service_patients else 0, 10)
    
    # Observations récentes (avec filtre de spécialité)
    observations_query = Observation.objects.select_related('patient', 'created_by')
    if selected_specialty:
        observations_query = observations_query.filter(theme_classe=selected_specialty)
    observations_recentes = observations_query[:5]
    
    # Activité des 7 derniers jours (avec filtre)
    seven_days_ago = timezone.now() - timedelta(days=7)
    observations_query_week = Observation.objects.filter(date__gte=seven_days_ago)
    if selected_specialty:
        observations_query_week = observations_query_week.filter(theme_classe=selected_specialty)
    observations_semaine = observations_query_week.count()
    
    nouveaux_patients_semaine = Patient.objects.filter(created_at__gte=seven_days_ago).count()
    
    # Liste des spécialités pour le filtre
    specialties_for_filter = [('', 'Toutes les spécialités')] + list(Observation.THEME_CHOICES)
    
    context = {
        'total_patients': total_filtered_patients if selected_specialty else total_patients,
        'total_observations': total_observations,
        'weekly_consultations': weekly_consultations,
        'max_consultations': max_consultations,
        'age_groups_with_percent': age_groups_with_percent,
        'service_patients': service_patients,
        'max_patients_service': max_patients_service,
        'observations_semaine': observations_semaine,
        'nouveaux_patients_semaine': nouveaux_patients_semaine,
        'observations_recentes': observations_recentes,
        'specialties_for_filter': specialties_for_filter,
        'selected_specialty': selected_specialty,
        'selected_specialty_display': dict(Observation.THEME_CHOICES).get(selected_specialty, 'Toutes les spécialités') if selected_specialty else 'Toutes les spécialités',
    }
    
    return render(request, 'med_assistant/dashboard.html', context)