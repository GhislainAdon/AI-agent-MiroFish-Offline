"""
Test du format de génération de profil pour vérifier la conformité avec les exigences OASIS
Vérification :
1. Le profil Twitter génère le format CSV
2. Le profil Reddit génère le format JSON détaillé
"""

import os
import sys
import json
import csv
import tempfile

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile


def test_profile_formats():
    """Test du format de profil"""
    print("=" * 60)
    print("Test du format de profil OASIS")
    print("=" * 60)
    
    # Créer des données de profil de test
    test_profiles = [
        OasisAgentProfile(
            user_id=0,
            user_name="test_user_123",
            name="Utilisateur Test",
            bio="Un utilisateur de test pour la validation",
            persona="L'Utilisateur Test est un participant enthousiaste dans les discussions sociales.",
            karma=1500,
            friend_count=100,
            follower_count=200,
            statuses_count=500,
            age=25,
            gender="male",
            mbti="INTJ",
            country="Chine",
            profession="Étudiant",
            interested_topics=["Technologie", "Éducation"],
            source_entity_uuid="test-uuid-123",
            source_entity_type="Étudiant",
        ),
        OasisAgentProfile(
            user_id=1,
            user_name="org_official_456",
            name="Organisation Officielle",
            bio="Compte officiel de l'Organisation",
            persona="Ceci est un compte institutionnel officiel qui communique les positions officielles.",
            karma=5000,
            friend_count=50,
            follower_count=10000,
            statuses_count=200,
            profession="Organisation",
            interested_topics=["Politique Publique", "Annonces"],
            source_entity_uuid="test-uuid-456",
            source_entity_type="Université",
        ),
    ]
    
    generator = OasisProfileGenerator.__new__(OasisProfileGenerator)
    
    # Utiliser un répertoire temporaire
    with tempfile.TemporaryDirectory() as temp_dir:
        twitter_path = os.path.join(temp_dir, "twitter_profiles.csv")
        reddit_path = os.path.join(temp_dir, "reddit_profiles.json")
        
        # Tester le format CSV de Twitter
        print("\n1. Test du profil Twitter (format CSV)")
        print("-" * 40)
        generator._save_twitter_csv(test_profiles, twitter_path)
        
        # Lire et vérifier le CSV
        with open(twitter_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        print(f"   Fichier : {twitter_path}")
        print(f"   Lignes : {len(rows)}")
        print(f"   En-têtes : {list(rows[0].keys())}")
        print(f"\n   Exemple de données (Ligne 1) :")
        for key, value in rows[0].items():
            print(f"     {key}: {value}")
        
        # Vérifier les champs requis
        required_twitter_fields = ['user_id', 'user_name', 'name', 'bio', 
                                   'friend_count', 'follower_count', 'statuses_count', 'created_at']
        missing = set(required_twitter_fields) - set(rows[0].keys())
        if missing:
            print(f"\n   [Erreur] Champs manquants : {missing}")
        else:
            print(f"\n   [Réussi] Tous les champs requis sont présents")
        
        # Tester le format JSON de Reddit
        print("\n2. Test du profil Reddit (format JSON détaillé)")
        print("-" * 40)
        generator._save_reddit_json(test_profiles, reddit_path)
        
        # Lire et vérifier le JSON
        with open(reddit_path, 'r', encoding='utf-8') as f:
            reddit_data = json.load(f)
        
        print(f"   Fichier : {reddit_path}")
        print(f"   Nombre d'entrées : {len(reddit_data)}")
        print(f"   Champs : {list(reddit_data[0].keys())}")
        print(f"\n   Exemple de données (Élément 1) :")
        print(json.dumps(reddit_data[0], ensure_ascii=False, indent=4))
        
        # Vérifier les champs du format détaillé
        required_reddit_fields = ['realname', 'username', 'bio', 'persona']
        optional_reddit_fields = ['age', 'gender', 'mbti', 'country', 'profession', 'interested_topics']
        
        missing = set(required_reddit_fields) - set(reddit_data[0].keys())
        if missing:
            print(f"\n   [Erreur] Champs requis manquants : {missing}")
        else:
            print(f"\n   [Réussi] Tous les champs requis sont présents")
        
        present_optional = set(optional_reddit_fields) & set(reddit_data[0].keys())
        print(f"   [Info] Champs optionnels : {present_optional}")
    
    print("\n" + "=" * 60)
    print("Test terminé !")
    print("=" * 60)


def show_expected_formats():
    """Afficher le format attendu par OASIS"""
    print("\n" + "=" * 60)
    print("Référence du format de profil attendu par OASIS")
    print("=" * 60)
    
    print("\n1. Profil Twitter (format CSV)")
    print("-" * 40)
    twitter_example = """user_id,user_name,name,bio,friend_count,follower_count,statuses_count,created_at
0,user0,User Zero,I am user zero with interests in technology.,100,150,500,2023-01-01
1,user1,User One,Tech enthusiast and coffee lover.,200,250,1000,2023-01-02"""
    print(twitter_example)
    
    print("\n2. Profil Reddit (format JSON détaillé)")
    print("-" * 40)
    reddit_example = [
        {
            "realname": "James Miller",
            "username": "millerhospitality",
            "bio": "Passionate about hospitality & tourism.",
            "persona": "James is a seasoned professional in the Hospitality & Tourism industry...",
            "age": 40,
            "gender": "male",
            "mbti": "ESTJ",
            "country": "UK",
            "profession": "Hospitality & Tourism",
            "interested_topics": ["Economics", "Business"]
        }
    ]
    print(json.dumps(reddit_example, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    test_profile_formats()
    show_expected_formats()
