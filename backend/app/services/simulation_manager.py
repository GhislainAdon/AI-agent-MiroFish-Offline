"""
Gestionnaire de simulation OASIS.
Gère les simulations parallèles sur les plateformes Twitter et Reddit.
Utilise des scripts prédéfinis et une génération intelligente des paramètres par LLM.
"""

import os
import json
import shutil
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityReader, FilteredEntities
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from .simulation_config_generator import SimulationConfigGenerator, SimulationParameters

logger = get_logger('mirofish.simulation')


class SimulationStatus(str, Enum):
    """Statut de la simulation"""
    CREATED = "created"
    PREPARING = "preparing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"      # Simulation arrêtée manuellement
    COMPLETED = "completed"  # Simulation terminée naturellement
    FAILED = "failed"


class PlatformType(str, Enum):
    """Type de plateforme"""
    TWITTER = "twitter"
    REDDIT = "reddit"


@dataclass
class SimulationState:
    """État de la simulation"""
    simulation_id: str
    project_id: str
    graph_id: str
    
    # État d'activation des plateformes
    enable_twitter: bool = True
    enable_reddit: bool = True
    
    # Statut
    status: SimulationStatus = SimulationStatus.CREATED
    
    # Données de la phase de préparation
    entities_count: int = 0
    profiles_count: int = 0
    entity_types: List[str] = field(default_factory=list)
    
    # Informations de génération de configuration
    config_generated: bool = False
    config_reasoning: str = ""
    
    # Données d'exécution
    current_round: int = 0
    twitter_status: str = "not_started"
    reddit_status: str = "not_started"
    
    # Horodatages
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Message d'erreur
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionnaire d'état complet (usage interne)"""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "enable_twitter": self.enable_twitter,
            "enable_reddit": self.enable_reddit,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "config_reasoning": self.config_reasoning,
            "current_round": self.current_round,
            "twitter_status": self.twitter_status,
            "reddit_status": self.reddit_status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
        }
    
    def to_simple_dict(self) -> Dict[str, Any]:
        """Dictionnaire d'état simplifié (usage retour API)"""
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "status": self.status.value,
            "entities_count": self.entities_count,
            "profiles_count": self.profiles_count,
            "entity_types": self.entity_types,
            "config_generated": self.config_generated,
            "error": self.error,
        }


class SimulationManager:
    """
    Gestionnaire de simulation.

    Fonctions principales :
    1. Lire et filtrer les entités depuis le graphe.
    2. Générer les profils agents OASIS.
    3. Générer par LLM les paramètres de configuration de simulation.
    4. Préparer tous les fichiers requis par les scripts prédéfinis.
    """
    
    # Répertoire de stockage des données de simulation
    SIMULATION_DATA_DIR = os.path.join(
        os.path.dirname(__file__), 
        '../../uploads/simulations'
    )
    
    def __init__(self):
        # Garantir l'existence du répertoire.
        os.makedirs(self.SIMULATION_DATA_DIR, exist_ok=True)
        
        # Cache mémoire des états de simulation.
        self._simulations: Dict[str, SimulationState] = {}
    
    def _get_simulation_dir(self, simulation_id: str) -> str:
        """Retourner le répertoire de données de simulation."""
        sim_dir = os.path.join(self.SIMULATION_DATA_DIR, simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        return sim_dir
    
    def _save_simulation_state(self, state: SimulationState):
        """Enregistrer l'état de simulation dans un fichier."""
        sim_dir = self._get_simulation_dir(state.simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        state.updated_at = datetime.now().isoformat()
        
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        
        self._simulations[state.simulation_id] = state
    
    def _load_simulation_state(self, simulation_id: str) -> Optional[SimulationState]:
        """Charger l'état de simulation depuis un fichier."""
        if simulation_id in self._simulations:
            return self._simulations[simulation_id]
        
        sim_dir = self._get_simulation_dir(simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        
        if not os.path.exists(state_file):
            return None
        
        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        state = SimulationState(
            simulation_id=simulation_id,
            project_id=data.get("project_id", ""),
            graph_id=data.get("graph_id", ""),
            enable_twitter=data.get("enable_twitter", True),
            enable_reddit=data.get("enable_reddit", True),
            status=SimulationStatus(data.get("status", "created")),
            entities_count=data.get("entities_count", 0),
            profiles_count=data.get("profiles_count", 0),
            entity_types=data.get("entity_types", []),
            config_generated=data.get("config_generated", False),
            config_reasoning=data.get("config_reasoning", ""),
            current_round=data.get("current_round", 0),
            twitter_status=data.get("twitter_status", "not_started"),
            reddit_status=data.get("reddit_status", "not_started"),
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            error=data.get("error"),
        )
        
        self._simulations[simulation_id] = state
        return state
    
    def create_simulation(
        self,
        project_id: str,
        graph_id: str,
        enable_twitter: bool = True,
        enable_reddit: bool = True,
    ) -> SimulationState:
        """
        Créer une nouvelle simulation.
        
        Args:
            project_id: ID du projet
            graph_id: ID du graphe
            enable_twitter: active ou non la simulation Twitter.
            enable_reddit: active ou non la simulation Reddit.
            
        Returns:
            SimulationState
        """
        import uuid
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"
        
        state = SimulationState(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            enable_twitter=enable_twitter,
            enable_reddit=enable_reddit,
            status=SimulationStatus.CREATED,
        )
        
        self._save_simulation_state(state)
        logger.info(f"Simulation créée : {simulation_id}, projet={project_id}, graphe={graph_id}")
        
        return state
    
    def prepare_simulation(
        self,
        simulation_id: str,
        simulation_requirement: str,
        document_text: str,
        defined_entity_types: Optional[List[str]] = None,
        use_llm_for_profiles: bool = True,
        progress_callback: Optional[callable] = None,
        parallel_profile_count: int = 3,
        storage: 'GraphStorage' = None,
    ) -> SimulationState:
        """
        Préparer l'environnement de simulation de façon entièrement automatisée.

        Étapes :
        1. Lire et filtrer les entités depuis le graphe.
        2. Générer un profil agent OASIS pour chaque entité.
        3. Générer par LLM les paramètres de simulation : temps, activité, fréquence de parole, etc.
        4. Enregistrer les fichiers de configuration et les profils.
        5. Laisser les scripts prédéfinis disponibles pour l'exécution.
        
        Args:
            simulation_id: ID de la simulation
            simulation_requirement: Description des exigences de simulation (pour la génération de configuration par LLM)
            document_text: Contenu du document original (pour la compréhension du contexte par LLM)
            defined_entity_types: Types d'entités prédéfinis (optionnel)
            use_llm_for_profiles: Utiliser ou non le LLM pour générer des profils détaillés
            progress_callback: Fonction de rappel de progression (étape, progression, message)
            parallel_profile_count: Nombre de générations de profils en parallèle, par défaut 3
            
        Returns:
            SimulationState
        """
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"La simulation n'existe pas : {simulation_id}")
        
        try:
            state.status = SimulationStatus.PREPARING
            self._save_simulation_state(state)
            
            sim_dir = self._get_simulation_dir(simulation_id)
            
            # ========== Phase 1 : lecture et filtrage des entités ==========
            if progress_callback:
                progress_callback("reading", 0, "Connexion au graphe...")

            if not storage:
                raise ValueError("storage (GraphStorage) est requis pour prepare_simulation")
            reader = EntityReader(storage)
            
            if progress_callback:
                progress_callback("reading", 30, "Lecture des données de nœuds...")
            
            filtered = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=defined_entity_types,
                enrich_with_edges=True
            )
            
            state.entities_count = filtered.filtered_count
            state.entity_types = list(filtered.entity_types)
            
            if progress_callback:
                progress_callback(
                    "reading", 100, 
                    f"Terminé, total {filtered.filtered_count} entités",
                    current=filtered.filtered_count,
                    total=filtered.filtered_count
                )
            
            if filtered.filtered_count == 0:
                state.status = SimulationStatus.FAILED
                state.error = "Aucune entité correspondant aux critères trouvée, vérifiez si le graphe est correctement construit"
                self._save_simulation_state(state)
                return state
            
            # ========== Phase 2 : génération des profils agents ==========
            total_entities = len(filtered.entities)
            
            if progress_callback:
                progress_callback(
                    "generating_profiles", 0, 
                    "Démarrage de la génération...",
                    current=0,
                    total=total_entities
                )
            
            # Transmettre graph_id pour enrichir les profils avec le contexte du graphe.
            generator = OasisProfileGenerator(storage=storage, graph_id=state.graph_id)
            
            def profile_progress(current, total, msg):
                if progress_callback:
                    progress_callback(
                        "generating_profiles", 
                        int(current / total * 100), 
                        msg,
                        current=current,
                        total=total,
                        item_name=msg
                    )
            
            # Définir le fichier d'enregistrement temps réel, avec préférence pour le JSON Reddit.
            realtime_output_path = None
            realtime_platform = "reddit"
            if state.enable_reddit:
                realtime_output_path = os.path.join(sim_dir, "reddit_profiles.json")
                realtime_platform = "reddit"
            elif state.enable_twitter:
                realtime_output_path = os.path.join(sim_dir, "twitter_profiles.csv")
                realtime_platform = "twitter"
            
            profiles = generator.generate_profiles_from_entities(
                entities=filtered.entities,
                use_llm=use_llm_for_profiles,
                progress_callback=profile_progress,
                graph_id=state.graph_id,  # Passer graph_id pour la recherche dans le graphe
                parallel_count=parallel_profile_count,  # Nombre de générations en parallèle
                realtime_output_path=realtime_output_path,  # Chemin de sauvegarde temps réel
                output_platform=realtime_platform  # Format de sortie
            )
            
            state.profiles_count = len(profiles)
            
            # Enregistrer les profils : Twitter utilise CSV, Reddit utilise JSON.
            # Reddit a déjà été enregistré en temps réel ; on sauvegarde à nouveau pour garantir la complétude.
            if progress_callback:
                progress_callback(
                    "generating_profiles", 95, 
                    "Sauvegarde des fichiers de profils...",
                    current=total_entities,
                    total=total_entities
                )
            
            if state.enable_reddit:
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "reddit_profiles.json"),
                    platform="reddit"
                )
            
            if state.enable_twitter:
                # Twitter utilise le format CSV : exigence OASIS.
                generator.save_profiles(
                    profiles=profiles,
                    file_path=os.path.join(sim_dir, "twitter_profiles.csv"),
                    platform="twitter"
                )
            
            if progress_callback:
                progress_callback(
                    "generating_profiles", 100, 
                    f"Terminé, total {len(profiles)} profils",
                    current=len(profiles),
                    total=len(profiles)
                )
            
            # ========== Phase 3 : génération intelligente de la configuration par LLM ==========
            if progress_callback:
                progress_callback(
                    "generating_config", 0, 
                    "Analyse des exigences de simulation...",
                    current=0,
                    total=3
                )
            
            config_generator = SimulationConfigGenerator()
            
            if progress_callback:
                progress_callback(
                    "generating_config", 30, 
                    "Appel au LLM pour générer la configuration...",
                    current=1,
                    total=3
                )
            
            sim_params = config_generator.generate_config(
                simulation_id=simulation_id,
                project_id=state.project_id,
                graph_id=state.graph_id,
                simulation_requirement=simulation_requirement,
                document_text=document_text,
                entities=filtered.entities,
                enable_twitter=state.enable_twitter,
                enable_reddit=state.enable_reddit
            )
            
            if progress_callback:
                progress_callback(
                    "generating_config", 70, 
                    "Sauvegarde des fichiers de configuration...",
                    current=2,
                    total=3
                )
            
            # Enregistrer les fichiers de configuration.
            config_path = os.path.join(sim_dir, "simulation_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(sim_params.to_json())
            
            state.config_generated = True
            state.config_reasoning = sim_params.generation_reasoning
            
            if progress_callback:
                progress_callback(
                    "generating_config", 100, 
                    "Génération de la configuration terminée",
                    current=3,
                    total=3
                )
            
            # Les scripts d'exécution restent dans backend/scripts/.
            # Au démarrage, simulation_runner exécute ces scripts depuis ce répertoire.
            
            # Mettre à jour le statut.
            state.status = SimulationStatus.READY
            self._save_simulation_state(state)
            
            logger.info(f"Préparation de la simulation terminée : {simulation_id}, "
                       f"entités={state.entities_count}, profils={state.profiles_count}")
            
            return state
            
        except Exception as e:
            logger.error(f"Échec de la préparation de la simulation : {simulation_id}, erreur={str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            state.status = SimulationStatus.FAILED
            state.error = str(e)
            self._save_simulation_state(state)
            raise
    
    def get_simulation(self, simulation_id: str) -> Optional[SimulationState]:
        """Récupérer l'état de simulation."""
        return self._load_simulation_state(simulation_id)
    
    def list_simulations(self, project_id: Optional[str] = None) -> List[SimulationState]:
        """Lister toutes les simulations."""
        simulations = []
        
        if os.path.exists(self.SIMULATION_DATA_DIR):
            for sim_id in os.listdir(self.SIMULATION_DATA_DIR):
                # Ignorer les fichiers cachés (comme .DS_Store) et les fichiers non-répertoires
                sim_path = os.path.join(self.SIMULATION_DATA_DIR, sim_id)
                if sim_id.startswith('.') or not os.path.isdir(sim_path):
                    continue
                
                state = self._load_simulation_state(sim_id)
                if state:
                    if project_id is None or state.project_id == project_id:
                        simulations.append(state)
        
        return simulations
    
    def get_profiles(self, simulation_id: str, platform: str = "reddit") -> List[Dict[str, Any]]:
        """Récupérer les profils agents d'une simulation."""
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"La simulation n'existe pas : {simulation_id}")
        
        sim_dir = self._get_simulation_dir(simulation_id)
        profile_path = os.path.join(sim_dir, f"{platform}_profiles.json")
        
        if not os.path.exists(profile_path):
            return []
        
        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_simulation_config(self, simulation_id: str) -> Optional[Dict[str, Any]]:
        """Récupérer la configuration de simulation."""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        
        if not os.path.exists(config_path):
            return None
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_run_instructions(self, simulation_id: str) -> Dict[str, str]:
        """Récupérer les instructions d'exécution."""
        sim_dir = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        scripts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../scripts'))
        
        return {
            "simulation_dir": sim_dir,
            "scripts_dir": scripts_dir,
            "config_file": config_path,
            "commands": {
                "twitter": f"python {scripts_dir}/run_twitter_simulation.py --config {config_path}",
                "reddit": f"python {scripts_dir}/run_reddit_simulation.py --config {config_path}",
                "parallel": f"python {scripts_dir}/run_parallel_simulation.py --config {config_path}",
            },
            "instructions": (
                f"1. Activer l'environnement conda : conda activate MiroFish\n"
                f"2. Exécuter la simulation (scripts situés dans {scripts_dir}) :\n"
                f"   - Exécuter Twitter seul : python {scripts_dir}/run_twitter_simulation.py --config {config_path}\n"
                f"   - Exécuter Reddit seul : python {scripts_dir}/run_reddit_simulation.py --config {config_path}\n"
                f"   - Exécuter les deux plateformes en parallèle : python {scripts_dir}/run_parallel_simulation.py --config {config_path}"
            )
        }
