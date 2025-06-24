from pathlib import Path
from datetime import datetime
from typing import List
from collections import defaultdict
import pandas as pd
import json

class SyntheaCSVSummarizer:
    #region init
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.csv_dir = self.data_dir / "csv"
        self.fhir_dir = self.data_dir / "fhir"

        self.patients = {}
        self.encounters = {}
        self.conditions = {}  # Nouveau: pour stocker les conditions/diagnostics
        
        print("Initialisé avec succès.")
    
    #endregion
    #region load_csv_data
    def load_csv_data(self):
        """Charger les données des fichiers CSV"""
        print("Chargement des données CSV...")
        
        # Charger les patients
        patients_file = self.csv_dir / "patients.csv"
        if patients_file.exists():
            patients_df = pd.read_csv(patients_file)
            for _, row in patients_df.iterrows():
                self.patients[row['Id']] = {
                    'name': f"{row.get('FIRST', '')} {row.get('LAST', '')}".strip(),
                    'gender': row.get('GENDER', ''),
                    'birthdate': row.get('BIRTHDATE', '')
                }
        
        # Charger les encounters
        encounters_file = self.csv_dir / "encounters.csv"
        if encounters_file.exists():
            encounters_df = pd.read_csv(encounters_file)
            for _, row in encounters_df.iterrows():
                self.encounters[row['Id']] = {
                    'patient_id': row.get('PATIENT', ''),
                    'provider_id': row.get('PROVIDER', ''),
                    'date': row.get('START', ''),
                    'encounter_class': row.get('ENCOUNTERCLASS', ''),
                    'description': row.get('DESCRIPTION', '')
                }
        
        # Charger les conditions/diagnostics
        conditions_file = self.csv_dir / "conditions.csv"
        if conditions_file.exists():
            conditions_df = pd.read_csv(conditions_file)
            for index, row in conditions_df.iterrows():
                self.conditions[index] = {
                    'patient_id': row.get('PATIENT', ''),
                    'encounter_id': row.get('ENCOUNTER', ''),
                    'start_date': row.get('START', ''),
                    'stop_date': row.get('STOP', ''),
                    'code': row.get('CODE', ''),
                    'description': row.get('DESCRIPTION', ''),
                    'system': row.get('SYSTEM', '')
                }
        
        print(f"Chargé: {len(self.patients)} patients, {len(self.encounters)} encounters, {len(self.conditions)} conditions")

    #endregion
    #region load_observations_csv
    def load_observations_csv(self):
        """Charger le fichier observations.csv"""
        observations_file = self.csv_dir / "observations.csv"
        
        if not observations_file.exists():
            print(f"Fichier {observations_file} non trouvé!")
            return pd.DataFrame()
        
        print("Chargement des observations CSV...")
        observations_df = pd.read_csv(observations_file)
        print(f"Chargé: {len(observations_df)} observations")
        
        return observations_df
    
    #endregion
    #region extract_fhir_diagnostics
    def extract_fhir_diagnostics(self):
        """Extraire les diagnostics des fichiers FHIR JSON"""
        print("Extraction des diagnostics FHIR...")
        
        fhir_diagnostics = {}  # encounter_id -> list of diagnostics
        
        for json_file in self.fhir_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    fhir_bundle = json.load(f)
                
                if 'entry' not in fhir_bundle:
                    continue
                
                for entry in fhir_bundle['entry']:
                    resource = entry.get('resource', {})
                    resource_type = resource.get('resourceType')
                    
                    match resource_type:
                        # Chercher les Condition (diagnostics)
                        case 'Condition':
                            encounter = resource.get('encounter', {})
                            if isinstance(encounter, dict):
                                encounter_ref = encounter.get('reference', '')
                                if encounter_ref.startswith('Encounter/'):
                                    encounter_id = encounter_ref.replace('Encounter/', '')
                                    
                                    # Extraire le diagnostic
                                    code = resource.get('code', {})
                                    if isinstance(code, dict):
                                        coding = code.get('coding', [])
                                        if isinstance(coding, list) and len(coding) > 0:
                                            condition_text = coding[0].get('display', '') if isinstance(coding[0], dict) else ''
                                            if condition_text:
                                                if encounter_id not in fhir_diagnostics:
                                                    fhir_diagnostics[encounter_id] = []
                                                fhir_diagnostics[encounter_id].append(condition_text)
                        
                        # Chercher les DiagnosticReport
                        case 'DiagnosticReport':
                            encounter = resource.get('encounter', {})
                            if isinstance(encounter, dict):
                                encounter_ref = encounter.get('reference', '')
                                if encounter_ref.startswith('Encounter/'):
                                    encounter_id = encounter_ref.replace('Encounter/', '')
                                    
                                    # Extraire le code du diagnostic
                                    code = resource.get('code', {})
                                    if isinstance(code, dict):
                                        coding = code.get('coding', [])
                                        if isinstance(coding, list) and len(coding) > 0:
                                            diagnostic_text = coding[0].get('display', '') if isinstance(coding[0], dict) else ''
                                            if diagnostic_text:
                                                if encounter_id not in fhir_diagnostics:
                                                    fhir_diagnostics[encounter_id] = []
                                                fhir_diagnostics[encounter_id].append(f"[Diagnostic] {diagnostic_text}")
                                    
                                    # Extraire la conclusion
                                    conclusion = resource.get('conclusion', '')
                                    if conclusion:
                                        if encounter_id not in fhir_diagnostics:
                                            fhir_diagnostics[encounter_id] = []
                                        fhir_diagnostics[encounter_id].append(f"[Conclusion] {conclusion}")
            
            except Exception as e:
                print(f"Erreur lors du traitement de {json_file}: {e}")
        
        print(f"Extrait {len(fhir_diagnostics)} entrées de diagnostics FHIR")
        return fhir_diagnostics
    
    #endregion
    #region get_csv_diagnostics_for_encounter
    def get_csv_diagnostics_for_encounter(self, encounter_id, date_str):
        """Récupérer les diagnostics CSV pour un encounter donné"""
        diagnostics = []
        
        # Convertir la date pour comparaison
        try:
            encounter_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
        except:
            encounter_date = None
        
        for condition_id, condition_data in self.conditions.items():
            if condition_data['encounter_id'] == encounter_id:
                # Vérifier si la condition est active à la date de l'encounter
                start_date = condition_data['start_date']
                stop_date = condition_data['stop_date']
                
                condition_active = True
                
                if start_date and encounter_date:
                    try:
                        condition_start = datetime.strptime(start_date[:10], '%Y-%m-%d')
                        if encounter_date < condition_start:
                            condition_active = False
                    except:
                        pass
                
                if stop_date and encounter_date:
                    try:
                        condition_stop = datetime.strptime(stop_date[:10], '%Y-%m-%d')
                        if encounter_date > condition_stop:
                            condition_active = False
                    except:
                        pass
                
                if condition_active:
                    description = condition_data['description']
                    if description:
                        diagnostics.append(description)
        
        return diagnostics
    
    #endregion
    #region group_observations
    def group_observations(self, observations_df, fhir_diagnostics):
        """Grouper les observations par patient, date et clinicien - Version vectorisée"""
        print("Groupement des observations...")
        
        # Vérifier que le DataFrame n'est pas vide
        if observations_df.empty:
            print("Aucune observation à grouper")
            return {}
        
        # Créer une copie de travail pour éviter les modifications du DataFrame original
        df_work = observations_df.copy()
        
        # Nettoyer et valider les données essentielles
        df_work = df_work.dropna(subset=['PATIENT'])  # Supprimer les lignes sans patient_id
        df_work['PATIENT'] = df_work['PATIENT'].astype(str)
        df_work['ENCOUNTER'] = df_work['ENCOUNTER'].fillna('').astype(str)
        
        if df_work.empty:
            print("Aucune observation valide après nettoyage")
            return {}
        
        # Formater les dates de manière vectorisée
        df_work['formatted_date'] = df_work['DATE'].apply(
            lambda x: self.format_date(x) if pd.notna(x) else ''
        )
        
        # Enrichir avec les informations des cliniciens de manière vectorisée
        def get_clinician_id(encounter_id):
            if not encounter_id:
                return 'prov_unknown'
            encounter_data = self.encounters.get(encounter_id)
            if encounter_data:
                if hasattr(encounter_data, 'get'):
                    return encounter_data.get('provider_id', 'prov_unknown')
                else:
                    return getattr(encounter_data, 'provider_id', 'prov_unknown')
            return 'prov_unknown'
        
        df_work['clinician_id'] = df_work['ENCOUNTER'].apply(get_clinician_id)
        
        # Formater les observations de manière vectorisée
        def format_obs_safe(row):
            try:
                return self.format_observation(row)
            except Exception as e:
                print(f"Erreur formatage observation {getattr(row, 'Id', 'unknown')}: {e}")
                return None
        
        df_work['observation_text'] = df_work.apply(format_obs_safe, axis=1)
        
        # Supprimer les observations qui n'ont pas pu être formatées
        df_work = df_work.dropna(subset=['observation_text'])
        df_work = df_work[df_work['observation_text'] != '']
        
        # Grouper par clé composite (patient, date, clinicien)
        grouped = df_work.groupby(['PATIENT', 'formatted_date', 'clinician_id'])
        
        # Construire le résultat final
        grouped_data = {}
        
        for group_key, group_df in grouped:
            patient_id, formatted_date, clinician_id = group_key
            
            # Construire l'entrée du groupe
            group_entry = {
                'patient_id': patient_id,
                'observations': group_df['observation_text'].tolist(),
                'diagnostics': [],  # Remplacé 'notes' par 'diagnostics'
                'date': formatted_date,
                'clinician_id': clinician_id
            }
            
            # Ajouter les diagnostics uniques pour ce groupe
            encounter_ids = group_df['ENCOUNTER'].dropna().unique()
            diagnostics_set = set()  # Pour éviter les doublons
            
            for encounter_id in encounter_ids:
                if encounter_id:
                    # Diagnostics des fichiers CSV
                    csv_diagnostics = self.get_csv_diagnostics_for_encounter(encounter_id, formatted_date)
                    for diagnostic in csv_diagnostics:
                        if diagnostic and diagnostic not in diagnostics_set:
                            diagnostics_set.add(diagnostic)
                            group_entry['diagnostics'].append(diagnostic)
                    
                    # Diagnostics des fichiers FHIR
                    if encounter_id in fhir_diagnostics:
                        fhir_diags = fhir_diagnostics[encounter_id]
                        for diagnostic in fhir_diags:
                            if diagnostic and diagnostic not in diagnostics_set:
                                diagnostics_set.add(diagnostic)
                                group_entry['diagnostics'].append(diagnostic)
            
            grouped_data[group_key] = group_entry
        
        print(f"Groupé en {len(grouped_data)} entrées uniques")
        return grouped_data
    #endregion

    #region create_summary_csv
    def create_summary_csv(self):
        """Créer le CSV résumé final avec groupement"""
        # Charger toutes les données
        self.load_csv_data()
        observations_df = self.load_observations_csv()
        fhir_diagnostics = self.extract_fhir_diagnostics()
        
        if observations_df.empty:
            print("Aucune observation trouvée!")
            return
        
        # Grouper les observations
        grouped_data = self.group_observations(observations_df, fhir_diagnostics)
        
        # Créer le résumé
        summary_data = []
        total_groups = len(grouped_data)
        
        print("Création du résumé...")
        for i, (group_key, group_info) in enumerate(grouped_data.items(), 1):
            print(f"Traitement du groupe {i}/{total_groups}...")
            
            try:
                # Concaténer les diagnostics avec ' : ' et terminer par '.'
                diagnostics_text = ' : '.join(group_info['diagnostics'])
                if diagnostics_text and not diagnostics_text.endswith('.'):
                    diagnostics_text += '.'

                # Concaténer les observations avec ' : ' et terminer par '.'
                observations_text = ' : '.join(group_info['observations'])
                if observations_text and not observations_text.endswith('.'):
                    observations_text += '.'
                
               
                
                summary_data.append({
                    'date': group_info['date'],
                    'diagnostics': diagnostics_text,  # Remplacé 'notes' par 'diagnostics'
                    'observations': observations_text,
                    'id_patient': group_info['patient_id'],
                    'id_clinician': group_info['clinician_id']
                })
                
            except Exception as e:
                print(f"Erreur lors du traitement du groupe {group_key}: {e}")
        
        # Créer le DataFrame et sauvegarder
        summary_df = pd.DataFrame(summary_data)
        
        # Trier par date
        summary_df['date_sort'] = pd.to_datetime(summary_df['date'], errors='coerce')
        summary_df = summary_df.sort_values('date_sort').drop('date_sort', axis=1)
        
        # Sauvegarder le CSV
        output_file = self.data_dir / "resume_observations_diagnostics.csv"
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\nRésumé créé avec succès!")
        print(f"Fichier: {output_file}")
        print(f"Nombre de lignes: {len(summary_df)}")
        print(f"Colonnes: {list(summary_df.columns)}")
        
        # Afficher un aperçu
        print("\nAperçu des premières lignes:")
        print(summary_df.head().to_string())
        
        # Générer les statistiques
        self.generate_stats(summary_df)
        
        return summary_df
    
    #endregion
    #region format_date
    def format_date(self, date_str):
        """Formater la date de manière lisible"""
        if not date_str:
            return ""
        
        try:
            # Essayer différents formats de date
            for fmt in ['%Y-%m-%d', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
            
            # Si aucun format ne marche, retourner tel quel
            return date_str[:10]  # Prendre juste les 10 premiers caractères
            
        except Exception:
            return date_str
    
    #endregion
    #region format_observation
    def format_observation(self, obs_row):
        """Formater le texte de l'observation en concaténant DESCRIPTION, VALUE et UNITS"""
        parts = []
        
        # Description principale
        description = obs_row.get('DESCRIPTION', '')
        if description and str(description) != 'nan':
            parts.append(str(description))
        
        # Valeur
        value = obs_row.get('VALUE', '')
        if value and str(value) != 'nan':
            parts.append(str(value))
        
        # Unités
        units = obs_row.get('UNITS', '')
        if units and str(units) != 'nan':
            parts.append(str(units))
        
        return ' '.join(parts) if parts else ''
    
    #endregion
    #region generate_stats
    def generate_stats(self, summary_df):
        """Générer des statistiques sur le résumé"""
        if summary_df.empty:
            return
        
        print(f"\n{'='*50}")
        print("STATISTIQUES DU RÉSUMÉ AVEC DIAGNOSTICS")
        print(f"{'='*50}")
        
        print(f"Total entrées groupées: {len(summary_df)}")
        print(f"Patients uniques: {summary_df['id_patient'].nunique()}")
        print(f"Cliniciens uniques: {summary_df['id_clinician'].nunique()}")
        
        # Période couverte
        dates = pd.to_datetime(summary_df['date'], errors='coerce').dropna()
        if not dates.empty:
            print(f"Période: {dates.min().strftime('%Y-%m-%d')} à {dates.max().strftime('%Y-%m-%d')}")
        
        # Top 5 des cliniciens les plus actifs
        print(f"\nTop 5 cliniciens:")
        clinician_counts = summary_df['id_clinician'].value_counts().head()
        for clinician, count in clinician_counts.items():
            print(f"  {clinician}: {count} entrées")
        
        # Entrées avec diagnostics
        with_diagnostics = summary_df[summary_df['diagnostics'].str.len() > 0]
        print(f"\nEntrées avec diagnostics: {len(with_diagnostics)} ({len(with_diagnostics)/len(summary_df)*100:.1f}%)")
        
        # Statistiques sur la longueur des résumés
        obs_lengths = summary_df['observations'].str.len()
        diagnostics_lengths = summary_df['diagnostics'].str.len()
        
        print(f"\nLongueur moyenne des observations: {obs_lengths.mean():.0f} caractères")
        print(f"Longueur moyenne des diagnostics: {diagnostics_lengths.mean():.0f} caractères")
        
        # Top 10 des diagnostics les plus fréquents
        print(f"\nTop 10 des diagnostics les plus fréquents:")
        all_diagnostics = []
        for diag_text in summary_df['diagnostics']:
            if diag_text:
                # Séparer les diagnostics multiples
                diagnostics = [d.strip() for d in diag_text.split(':') if d.strip()]
                all_diagnostics.extend(diagnostics)
        
        if all_diagnostics:
            from collections import Counter
            diag_counts = Counter(all_diagnostics)
            for diag, count in diag_counts.most_common(10):
                print(f"  {diag}: {count} occurrences")

    #endregion

