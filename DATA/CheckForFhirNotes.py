import json
import re
from pathlib import Path
from datetime import datetime

class SyntheaValidator:
    """Classe pour valider que les notes FHIR proviennent de Synthea"""
    
    def __init__(self):
        # Patterns typiques de Synthea
        self.synthea_patterns = {
            'patient_ids': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',  # UUID format
            'organization_names': ['Synthea', 'synthetic', 'test'],
            'common_names': ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Mary'],  # Noms fréquents
            'zip_codes': [r'^\d{5}$', r'^\d{5}-\d{4}$'],  # Formats ZIP US
            'phone_patterns': [r'^\d{3}-\d{3}-\d{4}$', r'^\(\d{3}\)\s?\d{3}-\d{4}$']
        }
    
    def validate_fhir_bundle(self, fhir_file_path):
        """Valide un fichier FHIR Bundle complet"""
        try:
            with open(fhir_file_path, 'r', encoding='utf-8') as f:
                bundle = json.load(f)
            
            validations = {
                'is_synthea': False,
                'bundle_structure': False,
                'patient_format': False,
                'organization_check': False,
                'timestamp_check': False,
                'resource_types': [],
                'synthea_indicators': []
            }
            
            # 1. Vérifier la structure Bundle
            if bundle.get('resourceType') == 'Bundle' and bundle.get('type') == 'transaction':
                validations['bundle_structure'] = True
                validations['synthea_indicators'].append('Bundle structure correcte')
            
            # 2. Analyser les entries
            entries = bundle.get('entry', [])
            for entry in entries:
                resource = entry.get('resource', {})
                resource_type = resource.get('resourceType')
                
                if resource_type:
                    validations['resource_types'].append(resource_type)
                
                # Vérifier le patient
                if resource_type == 'Patient':
                    validations.update(self._validate_patient(resource))
                
                # Vérifier l'organisation
                elif resource_type == 'Organization':
                    if self._validate_organization(resource):
                        validations['organization_check'] = True
                        validations['synthea_indicators'].append('Organisation Synthea détectée')
                
                # Vérifier les notes (DocumentReference)
                elif resource_type == 'DocumentReference':
                    note_validation = self._validate_document_reference(resource)
                    if note_validation:
                        validations['synthea_indicators'].extend(note_validation)
            
            # 3. Score final
            indicators = len(validations['synthea_indicators'])
            validations['is_synthea'] = indicators >= 2
            
            return validations
            
        except Exception as e:
            return {'error': f'Erreur lors de la validation: {str(e)}'}
    
    def _validate_patient(self, patient_resource):
        """Valide les caractéristiques d'un patient Synthea"""
        validations = {
            'patient_format': False,
            'synthea_indicators': []
        }
        
        # Vérifier l'ID UUID
        patient_id = patient_resource.get('id', '')
        if re.match(self.synthea_patterns['patient_ids'], patient_id):
            validations['patient_format'] = True
            validations['synthea_indicators'].append('ID patient au format UUID')
        
        # Vérifier les noms typiques
        names = patient_resource.get('name', [])
        for name in names:
            given_names = name.get('given', [])
            for given_name in given_names:
                if given_name in self.synthea_patterns['common_names']:
                    validations['synthea_indicators'].append(f'Prénom typique: {given_name}')
                    break
        
        # Vérifier les adresses (codes postaux US)
        addresses = patient_resource.get('address', [])
        for address in addresses:
            postal_code = address.get('postalCode', '')
            for pattern in self.synthea_patterns['zip_codes']:
                if re.match(pattern, postal_code):
                    validations['synthea_indicators'].append('Code postal US détecté')
                    break
        
        return validations
    
    def _validate_organization(self, org_resource):
        """Valide si l'organisation est typique de Synthea"""
        org_name = org_resource.get('name', '').lower()
        
        for synthea_term in self.synthea_patterns['organization_names']:
            if synthea_term in org_name:
                return True
        
        return False
    
    def _validate_document_reference(self, doc_resource):
        """Valide les notes DocumentReference"""
        indicators = []
        
        # Vérifier les auteurs
        authors = doc_resource.get('author', [])
        for author in authors:
            if author.get('display') and 'synthea' in author.get('display', '').lower():
                indicators.append('Auteur Synthea dans DocumentReference')
        
        # Vérifier le contenu
        content = doc_resource.get('content', [])
        for item in content:
            attachment = item.get('attachment', {})
            data = attachment.get('data')
            if data:
                # Décoder le base64 et chercher des patterns
                try:
                    import base64
                    decoded = base64.b64decode(data).decode('utf-8')
                    if any(term in decoded.lower() for term in ['synthea', 'synthetic', 'generated']):
                        indicators.append('Contenu avec références Synthea')
                except:
                    pass
        
        return indicators
    
    def validate_notes_collection(self, fhir_notes_dict):
        """Valide une collection de notes FHIR"""
        validation_report = {
            'total_notes': len(fhir_notes_dict),
            'synthea_indicators': 0,
            'suspicious_patterns': [],
            'is_likely_synthea': False
        }
        
        for encounter_id, note_content in fhir_notes_dict.items():
            # Vérifier les patterns dans le contenu
            if isinstance(note_content, str):
                note_lower = note_content.lower()
                
                # Mots-clés Synthea
                synthea_keywords = ['synthea', 'synthetic', 'generated', 'simulated']
                for keyword in synthea_keywords:
                    if keyword in note_lower:
                        validation_report['synthea_indicators'] += 1
                        validation_report['suspicious_patterns'].append(f'Keyword "{keyword}" found')
                        break
                
                # Patterns de dates artificiels (dates très récentes ou patterns répétitifs)
                date_patterns = re.findall(r'\d{4}-\d{2}-\d{2}', note_content)
                if len(set(date_patterns)) < len(date_patterns) * 0.5 and len(date_patterns) > 3:
                    validation_report['suspicious_patterns'].append('Dates répétitives détectées')
                
                # Patterns de noms répétitifs
                for common_name in self.synthea_patterns['common_names']:
                    if note_lower.count(common_name.lower()) > 1:
                        validation_report['suspicious_patterns'].append(f'Nom répétitif: {common_name}')
        
        # Évaluation finale
        ratio = validation_report['synthea_indicators'] / max(validation_report['total_notes'], 1)
        validation_report['is_likely_synthea'] = ratio > 0.1 or len(validation_report['suspicious_patterns']) > 5
        
        return validation_report


# Fonctions utilitaires pour usage simple
def quick_validate_fhir_file(file_path):
    """Validation rapide d'un fichier FHIR"""
    validator = SyntheaValidator()
    return validator.validate_fhir_bundle(file_path)

def quick_validate_notes(fhir_notes_dict):
    """Validation rapide d'une collection de notes"""
    validator = SyntheaValidator()
    return validator.validate_notes_collection(fhir_notes_dict)

# Exemple d'utilisation
if __name__ == "__main__":
    # Valider un fichier FHIR
    # result = quick_validate_fhir_file('path/to/your/fhir_file.json')
    # print(f"Est Synthea: {result['is_synthea']}")
    # print(f"Indicateurs: {result['synthea_indicators']}")
    
    # Valider des notes
    # notes = {'encounter1': 'Patient John Doe...', 'encounter2': 'Generated by Synthea...'}
    # result = quick_validate_notes(notes)
    # print(f"Probablement Synthea: {result['is_likely_synthea']}")
    pass
