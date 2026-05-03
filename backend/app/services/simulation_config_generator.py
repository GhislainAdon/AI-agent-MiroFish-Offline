"""
Générateur intelligent de configuration de simulation
Utiliser le LLM pour générer automatiquement des paramètres de simulation détaillés en fonction des exigences de simulation, du contenu des documents et des informations du graphe de connaissances
Implémenter l'automatisation complète du processus sans paramétrage manuel

Adopter une stratégie de génération par étapes pour éviter les échecs dus à la génération de contenu trop long d'un coup :
1. Générer la configuration temporelle
2. Générer la configuration d'événements
3. Générer les configurations d'agents par lots
4. Générer la configuration de plateforme
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityNode

logger = get_logger('mirofish.simulation_config')

# Configuration de fuseau horaire pour les horaires de travail chinois (Heure de Pékin)
CHINA_TIMEZONE_CONFIG = {
    # Heures creuses (presque aucune activité)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Heures matinales (réveil progressif)
    "morning_hours": [6, 7, 8],
    # Heures de travail
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Heures de pointe du soir (les plus actives)
    "peak_hours": [19, 20, 21, 22],
    # Heures nocturnes (l'activité diminue)
    "night_hours": [23],
    # Multiplicateurs d'activité
    "activity_multipliers": {
        "dead": 0.05,      # Presque personne au petit matin
        "morning": 0.4,    # Activité progressive le matin
        "work": 0.7,       # Activité moyenne pendant les heures de travail
        "peak": 1.5,       # Pointe du soir
        "night": 0.5       # L'activité diminue la nuit
    }
}


@dataclass
class AgentActivityConfig:
    """Configuration d'activité pour un agent unique"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Configuration d'activité (0.0-1.0)
    activity_level: float = 0.5  # Niveau d'activité global

    # Fréquence de parole (publications attendues par heure)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Périodes d'activité (format 24 heures, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # Vitesse de réponse (délai de réaction aux événements tendance, unité : minutes de simulation)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Tendance de sentiment (-1.0 à 1.0, négatif à positif)
    sentiment_bias: float = 0.0

    # Position (attitude envers des sujets spécifiques)
    stance: str = "neutral"  # soutien, opposition, neutre, observateur

    # Poids d'influence (détermine la probabilité que leur discours soit vu par d'autres agents)
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Configuration de simulation temporelle (basée sur les habitudes de travail chinoises)"""
    # Temps total de simulation (heures de simulation)
    total_simulation_hours: int = 72  # Par défaut 72 heures (3 jours)

    # Temps représenté par tour (minutes de simulation) - par défaut 60 minutes (1 heure), accélérer le temps
    minutes_per_round: int = 60

    # Plage d'agents activés par heure
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Heures de pointe (soir 19-22, période la plus active pour les Chinois)
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Heures creuses (petit matin 0-5, presque aucune activité)
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # Très faible activité au petit matin

    # Heures matinales
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    # Heures de travail
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Configuration d'événements"""
    # Publications initiales (déclenchant des événements au début de la simulation)
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # Événements planifiés (événements déclenchés à des moments spécifiques)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Mots-clés de sujets chauds
    hot_topics: List[str] = field(default_factory=list)

    # Direction narrative de l'opinion
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Configuration spécifique à la plateforme"""
    platform: str  # twitter ou reddit

    # Poids de l'algorithme de recommandation
    recency_weight: float = 0.4  # Fraîcheur temporelle
    popularity_weight: float = 0.3  # Popularité
    relevance_weight: float = 0.3  # Pertinence

    # Seuil de viralité (nombre d'interactions avant de déclencher la diffusion)
    viral_threshold: int = 10

    # Force de l'effet de chambre d'écho (degré de regroupement d'opinions similaires)
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Configuration complète des paramètres de simulation"""
    # Informations de base
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Configuration temporelle
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Liste de configurations d'agents
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Configuration d'événements
    event_config: EventConfig = field(default_factory=EventConfig)

    # Configuration de plateforme
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None

    # Configuration LLM
    llm_model: str = ""
    llm_base_url: str = ""

    # Métadonnées de génération
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # Explication du raisonnement du LLM

    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convertir en chaîne JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Générateur intelligent de configuration de simulation

    Utiliser le LLM pour analyser les exigences de simulation, le contenu des documents, les informations d'entités du graphe de connaissances,
    et générer automatiquement la configuration optimale des paramètres de simulation

    Adopter une stratégie de génération par étapes :
    1. Générer la configuration temporelle et la configuration d'événements (léger)
    2. Générer les configurations d'agents par lots (10-20 par lot)
    3. Générer la configuration de plateforme
    """

    # Longueur maximale du contexte en caractères
    MAX_CONTEXT_LENGTH = 50000
    # Nombre d'agents par lot
    AGENTS_PER_BATCH = 15

    # Longueur de troncature du contexte pour chaque étape (caractères)
    TIME_CONFIG_CONTEXT_LENGTH = 10000   # Configuration temporelle
    EVENT_CONFIG_CONTEXT_LENGTH = 8000   # Configuration d'événements
    ENTITY_SUMMARY_LENGTH = 300          # Résumé de l'entité
    AGENT_SUMMARY_LENGTH = 300           # Résumé de l'entité dans la configuration d'agent
    ENTITIES_PER_TYPE_DISPLAY = 20       # Nombre d'entités à afficher par type

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY non configuré")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        Générer intelligemment la configuration de simulation complète (génération par étapes)

        Args:
            simulation_id: ID de simulation
            project_id: ID de projet
            graph_id: ID du graphe de connaissances
            simulation_requirement: Description des exigences de simulation
            document_text: Contenu original du document
            entities: Liste d'entités filtrées
            enable_twitter: Indique s'il faut activer Twitter
            enable_reddit: Indique s'il faut activer Reddit
            progress_callback: Fonction de rappel de progression(étape_actuelle, étapes_totales, message)

        Returns:
            SimulationParameters: Paramètres de simulation complets
        """
        logger.info(f"Début de la génération intelligente de la configuration de simulation : simulation_id={simulation_id}, entités={len(entities)}")
        
        # Calculer le nombre total d'étapes
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # config temporelle + config événements + N lots d'agents + config plateforme
        current_step = 0

        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")

        # 1. Construire les informations contextuelles de base
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        # ========== Étape 1 : Générer la configuration temporelle ==========
        report_progress(1, "Génération de la configuration temporelle...")
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"Config temporelle : {time_config_result.get('reasoning', 'Succès')}")

        # ========== Étape 2 : Générer la configuration d'événements ==========
        report_progress(2, "Génération de la configuration d'événements et des sujets chauds...")
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"Config événements : {event_config_result.get('reasoning', 'Succès')}")

        # ========== Étapes 3-N : Générer les configurations d'agents par lots ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]

            report_progress(
                3 + batch_idx,
                f"Génération de la configuration d'agents ({start_idx + 1}-{end_idx}/{len(entities)})..."
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(f"Config agents : {len(all_agent_configs)} générées avec succès")

        # ========== Attribuer les agents de publication initiaux ==========
        logger.info("Attribution des agents de publication appropriés aux publications initiales...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(f"Publications initiales attribuées : {assigned_count} publications ont reçu un éditeur")

        # ========== Étape finale : Générer la configuration de plateforme ==========
        report_progress(total_steps, "Génération de la configuration de plateforme...")
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        # Construire les paramètres finaux
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"Génération de la configuration de simulation terminée : {len(params.agent_configs)} configurations d'agents")

        return params

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """Construire le contexte LLM, tronquer à la longueur maximale"""

        # Résumé de l'entité
        entity_summary = self._summarize_entities(entities)

        # Construire le contexte
        context_parts = [
            f"## Exigences de simulation\n{simulation_requirement}",
            f"\n## Informations sur les entités ({len(entities)})\n{entity_summary}",
        ]

        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  # Réserver 500 caractères

        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(document tronqué)"
            context_parts.append(f"\n## Contenu original du document\n{doc_text}")

        return "\n".join(context_parts)

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Générer le résumé des entités"""
        lines = []

        # Regrouper par type
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Inconnu"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)

        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)})")
            # Utiliser la quantité d'affichage et la longueur de résumé configurées
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... et {len(type_entities) - display_count} de plus")

        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """Appel LLM avec tentatives, incluant la logique de réparation JSON"""
        import re

        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # Baisser la température à chaque tentative
                    # Ne pas définir max_tokens, laisser le LLM générer librement
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                # Vérifier si la sortie a été tronquée
                if finish_reason == 'length':
                    logger.warning(f"Sortie du LLM tronquée (tentative {attempt+1})")
                    content = self._fix_truncated_json(content)

                # Essayer d'analyser le JSON
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"Échec de l'analyse JSON (tentative {attempt+1}) : {str(e)[:80]}")

                    # Essayer de réparer le JSON
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed

                    last_error = e

            except Exception as e:
                logger.warning(f"Échec de l'appel LLM (tentative {attempt+1}) : {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))

        raise last_error or Exception("Échec de l'appel LLM")
    
    def _fix_truncated_json(self, content: str) -> str:
        """Réparer le JSON tronqué"""
        content = content.strip()

        # Compter les parenthèses non fermées
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')

        # Vérifier les chaînes non fermées
        if content and content[-1] not in '",}]':
            content += '"'

        # Fermer les parenthèses
        content += ']' * open_brackets
        content += '}' * open_braces

        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Essayer de réparer le JSON de configuration"""
        import re

        # Réparer le cas tronqué
        content = self._fix_truncated_json(content)

        # Extraire la portion JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            # Supprimer les sauts de ligne dans les chaînes
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s

            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)

            try:
                return json.loads(json_str)
            except:
                # Essayer de supprimer tous les caractères de contrôle
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass

        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Générer la configuration temporelle"""
        # Utiliser la longueur de troncature de contexte configurée
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]

        # Calculer la valeur maximale autorisée (90 % des agents)
        max_agents_allowed = max(1, int(num_entities * 0.9))

        prompt = f"""Sur la base des exigences de simulation suivantes, générez la configuration de simulation temporelle.

{context_truncated}

## Tâche
Veuillez générer la configuration temporelle JSON.

### Principes de base (pour référence uniquement, ajuster flexiblement en fonction de la nature de l'événement et des caractéristiques des participants) :
- La base d'utilisateurs est chinoise, doit suivre les habitudes de travail de l'heure de Pékin
- 0-5h presque aucune activité (coefficient d'activité 0.05)
- 6-8h activité progressive (coefficient d'activité 0.4)
- 9-18 heures de travail activité modérée (coefficient d'activité 0.7)
- 19-22 soirée est la période de pointe (coefficient d'activité 1.5)
- Après 23h l'activité diminue (coefficient d'activité 0.5)
- Règle générale : faible activité au petit matin, augmentation progressive le matin, activité modérée pendant le travail, pointe le soir
- **Important** : Les valeurs d'exemple ci-dessous sont pour référence uniquement, ajuster les périodes spécifiques en fonction de la nature de l'événement et des caractéristiques des participants
  - Exemple : le pic étudiant peut être 21-23 ; les médias actifs toute la journée ; les institutions officielles uniquement pendant les heures de travail
  - Exemple : les dernières nouvelles peuvent provoquer des discussions tardives, off_peak_hours peuvent être raccourcies de manière appropriée

### Retourner au format JSON (pas de markdown)

Example:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Explication de la configuration temporelle pour cet événement"
}}

Description des champs :
- total_simulation_hours (int) : Temps total de simulation, 24-168 heures, court pour les dernières nouvelles, long pour les sujets en cours
- minutes_per_round (int) : Temps par tour, 30-120 minutes, recommandé 60 minutes
- agents_per_hour_min (int) : Nombre minimum d'agents activés par heure (plage : 1-{max_agents_allowed})
- agents_per_hour_max (int) : Nombre maximum d'agents activés par heure (plage : 1-{max_agents_allowed})
- peak_hours (tableau d'entiers) : Heures de pointe, ajuster en fonction des participants à l'événement
- off_peak_hours (tableau d'entiers) : Heures creuses, généralement tard le nuit/petit matin
- morning_hours (tableau d'entiers) : Heures matinales
- work_hours (tableau d'entiers) : Heures de travail
- reasoning (chaîne) : Brève explication pour cette configuration"""

        system_prompt = "Vous êtes un expert en simulation de médias sociaux. Retournez au format JSON pur, la configuration temporelle doit suivre les habitudes de travail chinoises."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Échec de la génération LLM de la config temporelle : {e}, utilisation de la configuration par défaut")
            return self._get_default_time_config(num_entities)
    
    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Obtenir la configuration temporelle par défaut (horaires de travail chinois)"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # 1 heure par tour, accélérer le temps
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Utilisation de la configuration par défaut des horaires de travail chinois (1 heure par tour)"
        }

    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """Analyser le résultat de la configuration temporelle et vérifier que agents_per_hour ne dépasse pas le nombre total d'agents"""
        # Obtenir les valeurs originales
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))

        # Vérifier et corriger : s'assurer de ne pas dépasser le nombre total d'agents
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) dépasse le nombre total d'agents ({num_entities}), corrigé")
            agents_per_hour_min = max(1, num_entities // 10)

        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) dépasse le nombre total d'agents ({num_entities}), corrigé")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)

        # S'assurer que min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max, corrigé à {agents_per_hour_min}")

        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),  # Par défaut 1 heure par tour
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # Presque personne au petit matin
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self,
        context: str,
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Générer la configuration d'événements"""

        # Obtenir la liste des types d'entités disponibles pour référence du LLM
        entity_types_available = list(set(
            e.get_entity_type() or "Inconnu" for e in entities
        ))

        # Lister les noms d'entités représentatifs pour chaque type
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Inconnu"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)

        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}"
            for t, examples in type_examples.items()
        ])

        # Utiliser la longueur de troncature de contexte configurée
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""Sur la base des exigences de simulation suivantes, générez la configuration d'événements.

Exigences de simulation : {simulation_requirement}

{context_truncated}

## Types d'entités disponibles et exemples
{type_info}

## Tâche
Veuillez générer la configuration d'événements JSON :
- Extraire les mots-clés de sujets chauds
- Décrire la direction de développement de l'opinion
- Concevoir le contenu des publications initiales, **chaque publication doit spécifier poster_type (type d'éditeur)**

**Important** : poster_type doit être sélectionné parmi les « Types d'entités disponibles » ci-dessus afin que les publications initiales puissent être attribuées aux agents appropriés pour la publication.
Exemple : Les déclarations officielles doivent être publiées par le type Official/University, les actualités par MediaOutlet, les opinions d'étudiants par le type Student.

Retourner au format JSON (pas de markdown) :
{{
    "hot_topics": ["mot-clé1", "mot-clé2", ...],
    "narrative_direction": "<description de la direction de développement de l'opinion>",
    "initial_posts": [
        {{"content": "contenu de la publication", "poster_type": "type d'entité (doit sélectionner parmi les types disponibles)"}},
        ...
    ],
    "reasoning": "<brève explication>"
}}"""

        system_prompt = "Vous êtes un expert en analyse d'opinions. Retournez au format JSON pur. Notez que poster_type doit correspondre précisément aux types d'entités disponibles."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"Échec de la génération LLM de la config événements : {e}, utilisation de la configuration par défaut")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Utilisation de la configuration par défaut"
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Analyser le résultat de la configuration d'événements"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Attribuer les agents de publication appropriés aux publications initiales

        Faire correspondre agent_id en fonction du poster_type de chaque publication
        """
        if not event_config.initial_posts:
            return event_config

        # Construire l'index des agents par type d'entité
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)

        # Table de mappage des types (gérer les différents formats que le LLM peut produire)
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }

        # Suivre les indices d'agents utilisés pour chaque type afin d'éviter de réutiliser le même agent
        used_indices: Dict[str, int] = {}

        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")

            # Essayer de trouver un agent correspondant
            matched_agent_id = None

            # 1. Correspondance directe
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. Correspondance en utilisant des alias
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break

            # 3. Si toujours pas trouvé, utiliser l'agent avec l'influence la plus élevée
            if matched_agent_id is None:
                logger.warning(f"Aucun agent correspondant trouvé pour le type '{poster_type}', utilisation de l'agent avec l'influence la plus élevée")
                if agent_configs:
                    # Trier par influence, sélectionner le plus élevé
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0

            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Inconnu"),
                "poster_agent_id": matched_agent_id
            })

            logger.info(f"Publication initiale attribuée : poster_type='{poster_type}' -> agent_id={matched_agent_id}")

        event_config.initial_posts = updated_posts
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """Générer les configurations d'agents par lots"""

        # Construire les informations d'entité (utiliser la longueur de résumé configurée)
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Inconnu",
                "summary": e.summary[:summary_len] if e.summary else ""
            })

        prompt = f"""Sur la base des informations suivantes, générez la configuration d'activité de médias sociaux pour chaque entité.

Exigences de simulation : {simulation_requirement}

## Liste des entités
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Tâche
Générer la configuration d'activité pour chaque entité, en notant :
- **Le temps suit les horaires de travail chinois** : Presque aucune activité 0-5h, plus actif 19-22h
- **Institutions officielles** (University/GovernmentAgency) : Faible activité (0.1-0.3), actives pendant les heures de travail (9-17), réponse lente (60-240 min), forte influence (2.5-3.0)
- **Médias** (MediaOutlet) : Activité moyenne (0.4-0.6), actifs toute la journée (8-23), réponse rapide (5-30 min), influence élevée (2.0-2.5)
- **Individus** (Student/Person/Alumni) : Forte activité (0.6-0.9), principalement actifs le soir (18-23), réponse rapide (1-15 min), faible influence (0.8-1.2)
- **Personnalités publiques/Experts** : Activité moyenne (0.4-0.6), influence moyenne-élevée (1.5-2.0)

Retourner au format JSON (pas de markdown) :
{{
    "agent_configs": [
        {{
            "agent_id": <doit correspondre à l'entrée>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <fréquence de publication>,
            "comments_per_hour": <fréquence de commentaire>,
            "active_hours": [<liste des heures actives, tenir compte des horaires de travail chinois>],
            "response_delay_min": <délai de réponse minimum en minutes>,
            "response_delay_max": <délai de réponse maximum en minutes>,
            "sentiment_bias": <-1.0 to 1.0>,
            "stance": "<soutien/opposition/neutre/observateur>",
            "influence_weight": <poids d'influence>
        }},
        ...
    ]
}}"""

        system_prompt = "Vous êtes un expert en analyse de comportement sur les médias sociaux. Retournez du JSON pur, la configuration doit suivre les habitudes de travail chinoises."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"Échec de la génération LLM par lot de la config agents : {e}, utilisation de la génération basée sur des règles")
            llm_configs = {}

        # Construire les objets AgentActivityConfig
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})

            # Si le LLM n'a pas généré, utiliser la génération basée sur des règles
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)

            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Inconnu",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)

        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Générer une configuration d'agent unique basée sur des règles (horaires de travail chinois)"""
        entity_type = (entity.get_entity_type() or "Inconnu").lower()

        if entity_type in ["university", "governmentagency", "ngo"]:
            # Institutions officielles : activité pendant les heures de travail, faible fréquence, forte influence
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9h00-17h59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            # Médias : activité toute la journée, fréquence moyenne, forte influence
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7h00-23h59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # Experts/Professeurs : activité travail + soir, fréquence moyenne
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8h00-21h59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # Étudiants : principalement le soir, forte fréquence
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # Matin + soir
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            # Anciens élèves : principalement le soir
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # Pause déjeuner + soir
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # Personnes ordinaires : pointe du soir
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # Journée + soir
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    

