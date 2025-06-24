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
        
        # Charger les encounters pour avoir le lien patient-praticien (avec ID du praticien seulement)
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
        
        print(f"Chargé: {len(self.patients)} patients, {len(self.encounters)} encounters")

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
    #region extract_fhir_notes
    def extract_fhir_notes(self):
        """Extraire les notes des praticiens des fichiers FHIR JSON"""
        print("Extraction des notes FHIR...")
        
        fhir_notes = {}  # encounter_id -> notes
        
        for json_file in self.fhir_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    fhir_bundle = json.load(f)
                
                if 'entry' not in fhir_bundle:
                    continue
                
                for entry in fhir_bundle['entry']:
                    resource = entry.get('resource', {})
                    resource_type = resource.get('resourceType')
                    
                    match resource_type :

                        # Chercher les DocumentReference (notes cliniques)
                        case 'DocumentReference':
                            # Vérifier si context existe et est un dictionnaire
                            context = resource.get('context', {})
                            if isinstance(context, dict):
                                encounter = context.get('encounter', {})
                                if isinstance(encounter, dict):
                                    encounter_ref = encounter.get('reference', '')
                                    if encounter_ref.startswith('Encounter/'):
                                        encounter_id = encounter_ref.replace('Encounter/', '')
                                        
                                        # Extraire le contenu de la note
                                        content = resource.get('content', [])
                                        if content and isinstance(content, list) and len(content) > 0:
                                            attachment = content[0].get('attachment', {}) if isinstance(content[0], dict) else {}
                                            if isinstance(attachment, dict):
                                                note_text = attachment.get('data', '')
                                                if note_text:
                                                    # Décoder base64 si nécessaire
                                                    try:
                                                        import base64
                                                        decoded_note = base64.b64decode(note_text).decode('utf-8')
                                                        fhir_notes[encounter_id] = decoded_note
                                                    except:
                                                        fhir_notes[encounter_id] = note_text
                        
                        # Chercher les DiagnosticReport avec des notes
                        case 'DiagnosticReport':
                            encounter = resource.get('encounter', {})
                            if isinstance(encounter, dict):
                                encounter_ref = encounter.get('reference', '')
                                if encounter_ref.startswith('Encounter/'):
                                    encounter_id = encounter_ref.replace('Encounter/', '')
                                    
                                    conclusion = resource.get('conclusion', '')
                                    if conclusion:
                                        if encounter_id in fhir_notes:
                                            fhir_notes[encounter_id] += f"\n[Diagnostic] {conclusion}"
                                        else:
                                            fhir_notes[encounter_id] = f"[Diagnostic] {conclusion}"
                    
                        # Chercher les Observation avec des notes
                        case 'Observation':
                            encounter = resource.get('encounter', {})
                            if isinstance(encounter, dict):
                                encounter_ref = encounter.get('reference', '')
                                if encounter_ref.startswith('Encounter/'):
                                    encounter_id = encounter_ref.replace('Encounter/', '')
                                    
                                    # Chercher des notes dans les composants
                                    components = resource.get('component', [])
                                    if isinstance(components, list):
                                        for comp in components:
                                            if isinstance(comp, dict) and 'note' in comp:
                                                notes = comp.get('note', [])
                                                if isinstance(notes, list) and len(notes) > 0:
                                                    note_text = notes[0].get('text', '') if isinstance(notes[0], dict) else ''
                                                    if note_text:
                                                        if encounter_id in fhir_notes:
                                                            fhir_notes[encounter_id] += f"\n{note_text}"
                                                        else:
                                                            fhir_notes[encounter_id] = note_text
                                
                                    # Chercher des notes directement dans l'observation
                                    notes = resource.get('note', [])
                                    if isinstance(notes, list):
                                        for note in notes:
                                            if isinstance(note, dict):
                                                note_text = note.get('text', '')
                                                if note_text:
                                                    if encounter_id in fhir_notes:
                                                        fhir_notes[encounter_id] += f"\n{note_text}"
                                                    else:
                                                        fhir_notes[encounter_id] = note_text
            
            except Exception as e:
                print(f"Erreur lors du traitement de {json_file}: {e}")
        
        print(f"Extrait {len(fhir_notes)} notes FHIR")
        return fhir_notes
    
    #endregion

    #region group_observations
    def group_observations(self, observations_df, fhir_notes):
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
                'notes': [],
                'date': formatted_date,
                'clinician_id': clinician_id
            }
            
            # Ajouter les notes FHIR uniques pour ce groupe
            encounter_ids = group_df['ENCOUNTER'].dropna().unique()
            notes_set = set()  # Pour éviter les doublons
            
            for encounter_id in encounter_ids:
                if encounter_id and encounter_id in fhir_notes:
                    clinician_notes = fhir_notes[encounter_id]
                    if clinician_notes and clinician_notes not in notes_set:
                        notes_set.add(clinician_notes)
                        group_entry['notes'].append(clinician_notes)
            
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
        fhir_notes = self.extract_fhir_notes()
        
        if observations_df.empty:
            print("Aucune observation trouvée!")
            return
        
        # Grouper les observations
        grouped_data = self.group_observations(observations_df, fhir_notes)
        
        # Créer le résumé
        summary_data = []
        total_groups = len(grouped_data)
        
        print("Création du résumé...")
        for i, (group_key, group_info) in enumerate(grouped_data.items(), 1):
            print(f"Traitement du groupe {i}/{total_groups}...")
            
            try:
                # Concaténer les observations avec ' : ' et terminer par '.'
                observations_text = ' : '.join(group_info['observations'])
                if observations_text and not observations_text.endswith('.'):
                    observations_text += '.'
                
                # Concaténer les notes avec ' : ' et terminer par '.'
                notes_text = ' : '.join(group_info['notes'])
                if notes_text and not notes_text.endswith('.'):
                    notes_text += '.'
                
                summary_data.append({
                    'date': group_info['date'],
                    'id_patient': group_info['patient_id'],
                    'id_clinician': group_info['clinician_id'],
                    'observations': observations_text,
                    'notes': notes_text
                })
                
            except Exception as e:
                print(f"Erreur lors du traitement du groupe {group_key}: {e}")
        
        # Créer le DataFrame et sauvegarder
        summary_df = pd.DataFrame(summary_data)
        
        # Trier par date
        summary_df['date_sort'] = pd.to_datetime(summary_df['date'], errors='coerce')
        summary_df = summary_df.sort_values('date_sort').drop('date_sort', axis=1)
        
        # Sauvegarder le CSV
        output_file = self.data_dir / "resume_observations_grouped.csv"
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
        print("STATISTIQUES DU RÉSUMÉ GROUPÉ")
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
        
        # Entrées avec notes
        with_notes = summary_df[summary_df['notes'].str.len() > 0]
        print(f"\nEntrées avec notes: {len(with_notes)} ({len(with_notes)/len(summary_df)*100:.1f}%)")
        
        # Statistiques sur la longueur des résumés
        obs_lengths = summary_df['observations'].str.len()
        notes_lengths = summary_df['notes'].str.len()
        
        print(f"\nLongueur moyenne des observations: {obs_lengths.mean():.0f} caractères")
        print(f"Longueur moyenne des notes: {notes_lengths.mean():.0f} caractères")

    #endregion

# Exemple d'utilisation
if __name__ == "__main__":
    # Initialiser
    summarizer = SyntheaCSVSummarizer(
        data_dir="path/to/synthea/data"
    )
    
    # Créer le résumé groupé
    summary_df = summarizer.create_summary_csv()