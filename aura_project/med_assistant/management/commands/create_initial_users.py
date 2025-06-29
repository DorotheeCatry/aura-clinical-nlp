"""
Commande Django pour créer les utilisateurs initiaux de l'hôpital
Usage: python manage.py create_initial_users
"""

from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from med_assistant.models import UserProfile
from django.db import transaction


class Command(BaseCommand):
    help = 'Crée les utilisateurs initiaux pour l\'hôpital AURA'

    def add_arguments(self, parser):
        parser.add_argument(
            '--force',
            action='store_true',
            help='Force la création même si des utilisateurs existent déjà',
        )

    def handle(self, *args, **options):
        # Vérifier si des utilisateurs existent déjà
        if User.objects.exists() and not options['force']:
            self.stdout.write(
                self.style.WARNING(
                    'Des utilisateurs existent déjà. Utilisez --force pour créer quand même.'
                )
            )
            return

        # Données des utilisateurs initiaux
        users_data = [
            {
                'username': 'dr.martin',
                'email': 'dr.martin@hopital-aura.fr',
                'first_name': 'Sophie',
                'last_name': 'Martin',
                'password': 'aura2024!',
                'is_staff': True,
                'is_superuser': True,
                'profile': {
                    'role': 'medecin_generaliste',
                    'specialite': 'Médecine générale',
                    'etablissement': 'Hôpital Central AURA',
                    'numero_ordre': '123456789'
                }
            },
            {
                'username': 'inf.dubois',
                'email': 'inf.dubois@hopital-aura.fr',
                'first_name': 'Marie',
                'last_name': 'Dubois',
                'password': 'aura2024!',
                'is_staff': False,
                'is_superuser': False,
                'profile': {
                    'role': 'infirmiere',
                    'specialite': 'Soins généraux',
                    'etablissement': 'Hôpital Central AURA',
                    'numero_ordre': 'INF987654321'
                }
            },
            {
                'username': 'psy.bernard',
                'email': 'psy.bernard@hopital-aura.fr',
                'first_name': 'Jean',
                'last_name': 'Bernard',
                'password': 'aura2024!',
                'is_staff': False,
                'is_superuser': False,
                'profile': {
                    'role': 'psychologue',
                    'specialite': 'Psychologie clinique',
                    'etablissement': 'Hôpital Central AURA',
                    'numero_ordre': 'PSY456789123'
                }
            }
        ]

        created_count = 0
        
        with transaction.atomic():
            for user_data in users_data:
                # Extraire les données du profil
                profile_data = user_data.pop('profile')
                
                # Créer ou récupérer l'utilisateur
                user, created = User.objects.get_or_create(
                    username=user_data['username'],
                    defaults=user_data
                )
                
                if created:
                    user.set_password(user_data['password'])
                    user.save()
                    created_count += 1
                    
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'✅ Utilisateur créé: {user.username} ({user.get_full_name()})'
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.WARNING(
                            f'⚠️  Utilisateur existe déjà: {user.username}'
                        )
                    )
                
                # Créer ou mettre à jour le profil
                profile, profile_created = UserProfile.objects.get_or_create(
                    user=user,
                    defaults=profile_data
                )
                
                if not profile_created:
                    # Mettre à jour le profil existant
                    for key, value in profile_data.items():
                        setattr(profile, key, value)
                    profile.save()
                    
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'🔄 Profil mis à jour: {profile.get_role_display()}'
                        )
                    )
                else:
                    self.stdout.write(
                        self.style.SUCCESS(
                            f'👤 Profil créé: {profile.get_role_display()}'
                        )
                    )

        # Résumé
        self.stdout.write('\n' + '='*50)
        self.stdout.write(
            self.style.SUCCESS(
                f'🏥 HÔPITAL AURA - Utilisateurs initialisés'
            )
        )
        self.stdout.write(
            self.style.SUCCESS(
                f'📊 {created_count} nouveaux utilisateurs créés'
            )
        )
        self.stdout.write('\n📋 Comptes disponibles:')
        
        for user_data in users_data:
            profile_data = user_data.get('profile', {})
            role_display = dict(UserProfile.ROLE_CHOICES).get(
                profile_data.get('role', ''), 
                'Rôle inconnu'
            )
            
            self.stdout.write(
                f'   👤 {user_data["username"]} / aura2024! - {role_display}'
            )
        
        self.stdout.write('\n🔐 Connexion: http://127.0.0.1:8000/aura/login/')
        self.stdout.write('='*50)