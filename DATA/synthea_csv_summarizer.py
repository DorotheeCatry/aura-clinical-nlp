import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
import re

class SyntheaCSVSummarizer:
    #region init
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.csv_dir = self.data_dir / "csv"
        self.fhir_dir = self.data_dir / "fhir"
        
        # Dictionnaires pour mapper les IDs
        self.patients = {}
        self.encounters = {}
    
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
                    
                    # Chercher les DocumentReference (notes cliniques)
                    if resource_type == 'DocumentReference':
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
                    elif resource_type == 'DiagnosticReport':
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
                    elif resource_type == 'Observation':
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
                # Optionnel : afficher plus de détails pour le débogage
                # import traceback
                # traceback.print_exc()
        
        print(f"Extrait {len(fhir_notes)} notes FHIR")
        return fhir_notes
    
    #endregion
    #region create_summary_csv
    def create_summary_csv(self):
        """Créer le CSV résumé final"""
        # Charger toutes les données
        self.load_csv_data()
        observations_df = self.load_observations_csv()
        fhir_notes = self.extract_fhir_notes()
        
        if observations_df.empty:
            print("Aucune observation trouvée!")
            return
        
        # Créer le résumé
        summary_data = []
        
        print("Création du résumé...")
        for _, obs in observations_df.iterrows():
            try:
                # Informations de base
                date = obs.get('DATE', '')
                patient_id = obs.get('PATIENT', '')
                encounter_id = obs.get('ENCOUNTER', '')
                
                # Formater la date
                formatted_date = self.format_date(date)
                
                # Nom du patient
                patient_name = "Patient inconnu"
                if patient_id in self.patients:
                    patient_name = self.patients[patient_id]['name']
                
                # ID du clinicien (via encounter)
                clinician_id = "Unknown"
                if encounter_id in self.encounters:
                    encounter = self.encounters[encounter_id]
                    provider_id = encounter.get('provider_id', '')
                    if provider_id:
                        clinician_id = provider_id
                
                # Observation
                observation_text = self.format_observation(obs)
                
                # Notes du clinicien
                clinician_notes = fhir_notes.get(encounter_id, '')
                
                summary_data.append({
                    'date': formatted_date,
                    'nom_patient': patient_name,
                    'id_clinicien': clinician_id,
                    'observations': observation_text,
                    'notes_clinicien': clinician_notes
                })
                
            except Exception as e:
                print(f"Erreur lors du traitement de l'observation {obs.get('Id', 'unknown')}: {e}")
        
        # Créer le DataFrame et sauvegarder
        summary_df = pd.DataFrame(summary_data)
        
        # Trier par date
        summary_df['date_sort'] = pd.to_datetime(summary_df['date'], errors='coerce')
        summary_df = summary_df.sort_values('date_sort').drop('date_sort', axis=1)
        
        # Sauvegarder le CSV
        output_file = self.data_dir / "resume_observations.csv"
        summary_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\nRésumé créé avec succès!")
        print(f"Fichier: {output_file}")
        print(f"Nombre de lignes: {len(summary_df)}")
        print(f"Colonnes: {list(summary_df.columns)}")
        
        # Afficher un aperçu
        print("\nAperçu des premières lignes:")
        print(summary_df.head().to_string())
        
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
        """Formater le texte de l'observation"""
        parts = []
        
        # Description principale
        description = obs_row.get('DESCRIPTION', '')
        if description:
            parts.append(description)
        
        # Valeur et unité
        value = obs_row.get('VALUE', '')
        units = obs_row.get('UNITS', '')
        
        if value and str(value) != 'nan':
            value_text = str(value)
            if units and str(units) != 'nan':
                value_text += f" {units}"
            parts.append(f"Valeur: {value_text}")
        
        # Type d'observation
        obs_type = obs_row.get('TYPE', '')
        if obs_type:
            parts.append(f"Type: {obs_type}")
        
        # Code
        code = obs_row.get('CODE', '')
        if code:
            parts.append(f"Code: {code}")
        
        return " | ".join(parts)
    
    #endregion
    #region generate_stats
    def generate_stats(self, summary_df):
        """Générer des statistiques sur le résumé"""
        if summary_df.empty:
            return
        
        print(f"\n{'='*50}")
        print("STATISTIQUES DU RÉSUMÉ")
        print(f"{'='*50}")
        
        print(f"Total observations: {len(summary_df)}")
        print(f"Patients uniques: {summary_df['nom_patient'].nunique()}")
        print(f"Cliniciens uniques: {summary_df['id_clinicien'].nunique()}")
        
        # Période couverte
        dates = pd.to_datetime(summary_df['date'], errors='coerce').dropna()
        if not dates.empty:
            print(f"Période: {dates.min().strftime('%Y-%m-%d')} à {dates.max().strftime('%Y-%m-%d')}")
        
        # Top 5 des cliniciens les plus actifs
        print(f"\nTop 5 cliniciens:")
        clinician_counts = summary_df['id_clinicien'].value_counts().head()
        for clinician, count in clinician_counts.items():
            print(f"  {clinician}: {count} observations")
        
        # Observations avec notes
        with_notes = summary_df[summary_df['notes_clinicien'].str.len() > 0]
        print(f"\nObservations avec notes: {len(with_notes)} ({len(with_notes)/len(summary_df)*100:.1f}%)")

    #endregion