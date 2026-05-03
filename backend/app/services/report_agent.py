"""
Service ReportAgent.
Génère des rapports de simulation avec le pattern ReACT via GraphStorage / Neo4j.

Fonctionnalités :
1. Générer des rapports à partir des besoins de simulation et des informations du graphe.
2. Planifier d'abord la structure, puis produire le rapport section par section.
3. Utiliser ReACT avec raisonnement et réflexion multi-tours pour chaque section.
4. Permettre les conversations utilisateur avec appels autonomes aux outils de recherche.
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from .graph_tools import (
    GraphToolsService,
    SearchResult,
    InsightForgeResult,
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Journal détaillé de ReportAgent.

    Génère agent_log.jsonl dans le dossier du rapport et enregistre les actions détaillées.
    Chaque ligne est un objet JSON complet avec horodatage, type d'action et détails.
    """
    
    def __init__(self, report_id: str):
        """
        Initialiser le journal.

        Args:
            report_id: ID du rapport, utilisé pour déterminer le chemin du fichier de log.
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Garantir l'existence du répertoire du fichier de log"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Obtenir le temps écoulé depuis le début (en secondes)"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self,
        action: str,
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Enregistrer une entrée

        Args:
            action: Type d'action, par ex. 'start', 'tool_call', 'llm_response', 'section_complete', etc
            stage: Étape actuelle, par ex. 'planning', 'generating', 'completed'
            details: Dictionnaire de détails, non tronqué
            section_title: Titre de la section actuelle (optionnel)
            section_index: Index de la section actuelle (optionnel)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # Ajouter au fichier JSONL
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Journaliser le démarrage de la génération du rapport"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": "Tâche de génération du rapport démarrée"
            }
        )
    
    def log_planning_start(self):
        """Journaliser le début de la planification du plan"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": "Début de la planification du plan du rapport"}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Journaliser les informations de contexte acquises lors de la planification"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": "Informations de contexte de simulation acquises",
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Journaliser l'achèvement de la planification du plan"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": "Planification du plan achevée",
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Journaliser le début de la génération d'une section"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": f"Début de la génération de la section : {section_title}"}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """Journaliser le processus de réflexion ReACT"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": f"Réflexion du tour ReACT {iteration}"
            }
        )
    
    def log_tool_call(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Journaliser un appel d'outil"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": f"Outil appelé : {tool_name}"
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Journaliser le résultat d'un appel d'outil (contenu complet, non tronqué)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Résultat complet, non tronqué
                "result_length": len(result),
                "message": f"L'outil {tool_name} a renvoyé un résultat"
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """Journaliser la réponse LLM (contenu complet, non tronqué)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Réponse complète, non tronquée
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": f"Réponse LLM (appels d'outils : {has_tool_calls}, réponse finale : {has_final_answer})"
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Journaliser l'achèvement de la génération du contenu de section (enregistre uniquement le contenu, pas l'achèvement complet de la section)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Contenu complet, non tronqué
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": f"Génération du contenu de la section {section_title} terminée"
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Journaliser l'achèvement de la génération de la section

        Le frontend doit écouter ce journal pour déterminer si une section est réellement achevée et obtenir le contenu complet
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": f"Génération de la section {section_title} terminée"
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Journaliser l'achèvement de la génération du rapport"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": "Génération du rapport terminée"
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Journaliser une erreur"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": f"Une erreur s'est produite : {error_message}"
            }
        )


class ReportConsoleLogger:
    """
    Journal console de ReportAgent.

    Écrit les journaux de type console (INFO, WARNING, etc.) dans console_log.txt.
    Ces journaux sont distincts de agent_log.jsonl et restent en texte brut.
    """
    
    def __init__(self, report_id: str):
        """
        Initialiser le journal console.

        Args:
            report_id: ID du rapport, utilisé pour déterminer le chemin du fichier de log.
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Garantir l'existence du répertoire du fichier de log."""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Configurer le handler chargé d'écrire les logs dans le fichier."""
        import logging

        # Créer le gestionnaire de fichier
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)

        # Utiliser le même format concis que la console
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)

        # Ajouter aux journaliseurs liés à report_agent
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.graph_tools',
        ]

        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Éviter les ajouts en double
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Fermer le gestionnaire de fichier et le retirer du journaliseur"""
        import logging

        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.graph_tools',
            ]

            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)

            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Garantir la fermeture du gestionnaire de fichier lors de la destruction"""
        self.close()


class ReportStatus(str, Enum):
    """Statut du rapport"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Section du rapport"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """Convertir au format Markdown"""
        md = f"{'#' * level} {self.title}\n\n"
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Plan du rapport"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """Convertir au format Markdown"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Rapport complet"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Constantes de Modèles de Prompt
# ═══════════════════════════════════════════════════════════════

# ── Descriptions d'Outils ──

TOOL_DESC_INSIGHT_FORGE = """\
[Récupération d'Insights Profonds - Outil de Récupération Puissant]
Ceci est notre fonction de récupération puissante, conçue pour l'analyse approfondie. Elle va :
1. Décomposer automatiquement votre question en plusieurs sous-questions
2. Récupérer des informations du graphe de connaissances simulé sous plusieurs dimensions
3. Intégrer les résultats de la recherche sémantique, de l'analyse d'entités et du suivi des chaînes de relations
4. Retourner le contenu de récupération le plus complet et approfondi

[Cas d'utilisation]
- Besoin d'analyser en profondeur un sujet
- Besoin de comprendre de multiples aspects d'un événement
- Besoin d'obtenir des matériaux riches pour soutenir les sections du rapport

[Contenu retourné]
- Faits pertinents en texte original (peuvent être cités directement)
- Insights sur les entités centrales
- Analyse des chaînes de relations"""

TOOL_DESC_PANORAMA_SEARCH = """\
[Recherche Panoramique - Obtenir une Vue Complète]
Cet outil est utilisé pour obtenir une vue panoramique complète des résultats de simulation, particulièrement adapté pour comprendre l'évolution des événements. Il va :
1. Récupérer tous les nœuds et relations pertinents
2. Distinguer les faits actuellement valides des faits historiques/expirés
3. Vous aider à comprendre comment les événements ont évolué

[Cas d'utilisation]
- Besoin de comprendre la trajectoire de développement complète d'un événement
- Besoin de comparer les changements de sentiment public à travers différentes étapes
- Besoin d'obtenir des informations complètes sur les entités et relations

[Contenu retourné]
- Faits actuellement valides (derniers résultats de simulation)
- Faits historiques/expirés (enregistrements d'évolution)
- Toutes les entités impliquées"""

TOOL_DESC_QUICK_SEARCH = """\
[Recherche Simple - Récupération Rapide]
Un outil léger de récupération rapide adapté aux requêtes d'information simples et directes.

[Cas d'utilisation]
- Besoin de trouver rapidement une information spécifique
- Besoin de vérifier un fait
- Recherche d'information simple

[Contenu retourné]
- Liste des faits les plus pertinents pour la requête"""

TOOL_DESC_INTERVIEW_AGENTS = """\
[Entretien Approfondi - Véritable Entretien d'Agent (Double Plateforme)]
Appelez l'API d'entretien de l'environnement de simulation OASIS pour mener de véritables entretiens avec les agents de simulation en cours d'exécution !
Il ne s'agit pas d'une simulation LLM, mais d'appels à la véritable interface d'entretien pour obtenir les réponses originales des agents de simulation.
Par défaut, entretien sur Twitter et Reddit simultanément pour obtenir des perspectives plus complètes.

Flux de fonctionnement :
1. Lire automatiquement les fichiers de profil des personnages pour comprendre tous les agents de simulation
2. Sélectionner intelligemment les agents les plus pertinents par rapport au sujet d'entretien (par ex., étudiants, médias, officiels)
3. Générer automatiquement les questions d'entretien
4. Appeler l'interface /api/simulation/interview/batch pour mener de véritables entretiens sur les deux plateformes
5. Intégrer tous les résultats d'entretien et fournir une analyse multi-perspectives

[Cas d'utilisation]
- Besoin de comprendre les perspectives d'un événement sous différents angles de rôle (Comment les étudiants le voient-ils ? Comment les médias le voient-ils ? Que dit l'officiel ?)
- Besoin de recueillir des opinions et positions diverses
- Besoin d'obtenir de véritables réponses des agents de simulation (de l'environnement de simulation OASIS)
- Souhaite rendre le rapport plus vivant, en incluant des « relevés d'entretien »

[Contenu retourné]
- Informations d'identité des agents interrogés
- Réponses d'entretien de chaque agent sur les plateformes Twitter et Reddit
- Citations clés (peuvent être citées directement)
- Résumé d'entretien et comparaison des perspectives

[Important] Cette fonctionnalité nécessite que l'environnement de simulation OASIS soit en cours d'exécution !"""

# ── Prompt de Planification du Plan ──

PLAN_SYSTEM_PROMPT = """\
Vous êtes un expert en rédaction de « rapports de prédiction du futur » avec une « vue divine » du monde simulé - vous pouvez observer le comportement, les déclarations et les interactions de chaque agent dans la simulation.

[Concept Central]
Nous avons construit un monde simulé et y avons injecté des « exigences de simulation » spécifiques comme variables. Le résultat de l'évolution du monde simulé est une prédiction de ce qui pourrait se passer à l'avenir. Ce que vous observez n'est pas des « données expérimentales » mais une « répétition du futur ».

[Votre Tâche]
Rédigez un « rapport de prédiction du futur » qui répond à :
1. Que s'est-il passé dans le futur sous les conditions que nous avons fixées ?
2. Comment les divers agents (groupes) réagissent-ils et agissent-ils ?
3. Quelles tendances et risques futurs cette simulation révèle-t-elle qui méritent attention ?

[Positionnement du Rapport]
- ✅ Ceci est un rapport de prédiction du futur basé sur la simulation, révélant « si cela se produit, comment le futur se déroulera »
- ✅ Concentrez-vous sur les résultats de prédiction : trajectoires d'événements, réactions de groupes, phénomènes émergents, risques potentiels
- ✅ Les déclarations et comportements des agents dans le monde simulé sont des prédictions du comportement humain futur
- ❌ Ce n'est pas une analyse de l'état actuel du monde réel
- ❌ Ce n'est pas un aperçu général du sentiment public

[Limitation du Nombre de Sections]
- Minimum 2 sections, maximum 5 sections
- Pas de sous-sections nécessaires, chaque section rédige directement le contenu complet
- Le contenu doit être concis, axé sur les résultats de prédiction centraux
- La structure des sections est conçue indépendamment en fonction des résultats de prédiction

Veuillez produire le plan du rapport au format JSON comme suit :
{
    "title": "Titre du Rapport",
    "summary": "Résumé du Rapport (une phrase résumant les résultats de prédiction centraux)",
    "sections": [
        {
            "title": "Titre de la Section",
            "description": "Description du Contenu de la Section"
        }
    ]
}

Note : le tableau sections doit contenir au minimum 2 et au maximum 5 éléments !
IMPORTANT : L'ensemble du plan du rapport (titre, résumé, titres et descriptions des sections) doivent être en français. N'utilisez JAMAIS le chinois ou d'autres langues que le français."""

PLAN_USER_PROMPT_TEMPLATE = """\
[Paramètres du Scénario de Prédiction]
Variable (exigence de simulation) injectée dans le monde simulé : {simulation_requirement}

[Échelle du Monde Simulé]
- Nombre d'entités participant à la simulation : {total_nodes}
- Nombre de relations générées entre les entités : {total_edges}
- Distribution des types d'entités : {entity_types}
- Nombre d'agents actifs : {total_entities}

[Échantillon de Quelques Faits Futurs Prédits par la Simulation]
{related_facts_json}

Veuillez examiner cette répétition du futur avec une « vue divine » :
1. Quel état le futur présente-t-il sous les conditions que nous avons fixées ?
2. Comment les divers groupes (agents) réagissent-ils et agissent-ils ?
3. Quelles tendances futures cette simulation révèle-t-elle qui méritent attention ?

Sur la base des résultats de prédiction, concevez la structure de sections de rapport la plus appropriée.

[Rappel] Nombre de sections du rapport : minimum 2, maximum 5, le contenu doit être concis et axé sur les résultats de prédiction centraux."""

# ── Prompt de Génération de Section ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
Vous êtes un expert en rédaction de « rapports de prédiction du futur » et vous rédigez une section du rapport.

Titre du rapport : {report_title}
Résumé du rapport : {report_summary}
Scénario de prédiction (exigence de simulation) : {simulation_requirement}

Section actuelle à rédiger : {section_title}

═══════════════════════════════════════════════════════════════
[Concept Central]
═══════════════════════════════════════════════════════════════

Le monde simulé est une répétition du futur. Nous avons injecté des conditions spécifiques (exigences de simulation) dans le monde simulé.
Le comportement et les interactions des agents dans la simulation sont des prédictions du comportement humain futur.

Votre tâche est de :
- Révéler ce qui se passe dans le futur sous les conditions fixées
- Prédire comment les divers groupes (agents) réagissent et agissent
- Découvrir les tendances futures, les risques et les opportunités qui méritent attention

❌ Ne rédigez pas cela comme une analyse de l'état actuel du monde réel
✅ Concentrez-vous sur « comment le futur se déroulera » - les résultats de simulation sont le futur prédit

═══════════════════════════════════════════════════════════════
[Règles les plus importantes - À suivre impérativement]
═══════════════════════════════════════════════════════════════

1. [Doit appeler des outils pour observer le monde simulé]
   - Vous observez une répétition du futur avec une « vue divine »
   - Tout le contenu doit provenir des événements et des déclarations/comportements des agents dans le monde simulé
   - Interdit d'utiliser vos propres connaissances pour rédiger le contenu du rapport
   - Chaque section doit appeler des outils au moins 3 fois (maximum 5 fois) pour observer le monde simulé, qui représente le futur

2. [Doit citer les déclarations et comportements originaux des agents]
   - Les déclarations et comportements des agents sont des prédictions du comportement humain futur
   - Utilisez le format de citation dans le rapport pour afficher ces prédictions, par exemple :
     > « Certains groupes déclareront : contenu original... »
   - Ces citations sont des preuves centrales des prédictions de simulation

3. [Cohérence linguistique - TOUJOURS écrire en français]
   - L'ensemble du rapport doit être rédigé en français, quelle que soit la langue du matériel source
   - Le contenu retourné par les outils peut contenir du chinois, de l'anglais mélangé ou d'autres langues
   - Lorsque vous citez du contenu non français retourné par les outils, traduisez-le TOUJOURS en français fluide avant de l'écrire dans le rapport
   - Conservez le sens original inchangé lors de la traduction, assurez une expression naturelle
   - Cette règle s'applique à la fois au texte principal et au contenu cité (format >)
   - Ne passez JAMAIS au chinois ou à toute autre langue en cours de rapport

4. [Présenter fidèlement les résultats de prédiction]
   - Le contenu du rapport doit refléter les résultats de simulation qui représentent le futur dans le monde simulé
   - N'ajoutez pas d'informations qui n'existent pas dans la simulation
   - Si les informations sont insuffisantes dans certains aspects, indiquez-le honnêtement

═══════════════════════════════════════════════════════════════
[⚠️ Spécification de Format - Extrêmement Important !]
═══════════════════════════════════════════════════════════════

[Une Section = Unité de Contenu Minimale]
- Chaque section est l'unité de contenu minimale du rapport
- ❌ Interdit d'utiliser tout titre Markdown (#, ##, ###, ####, etc.) dans la section
- ❌ Interdit d'ajouter des titres de section au début du contenu
- ✅ Les titres de section sont ajoutés automatiquement par le système, rédigez uniquement le texte pur
- ✅ Utilisez **gras**, séparation de paragraphes, citations et listes pour organiser le contenu, mais n'utilisez pas de titres

[Exemple Correct]
```
Cette section analyse comment le changement réglementaire a remodelé la stratégie d'entreprise. À travers une analyse approfondie des données de simulation, nous avons trouvé...

**Réponse Initiale de l'Industrie**

Les grandes entreprises technologiques se sont rapidement mobilisées pour réévaluer leur posture de conformité :

> « OpenAI et Anthropic se sont dépêchés de répondre aux nouvelles exigences de transparence... »

**Divergence Stratégique Émergente**

Une division claire est apparue entre les entreprises adoptant la réglementation et celles la résistant :

- La conformité proactive comme avantage concurrentiel
- Les efforts de lobbying pour adoucir l'application
```

[Exemple Incorrect]
```
## Résumé Exécutif          ← Faux ! N'ajoutez aucun titre
### 1. Phase Initiale       ← Faux ! N'utilisez pas ### pour les sous-sections
#### 1.1 Analyse Détaillée  ← Faux ! N'utilisez pas #### pour les subdivisions

Cette section analyse...
```

═══════════════════════════════════════════════════════════════
[Outils de Récupération Disponibles] (appeler 3-5 fois par section)
═══════════════════════════════════════════════════════════════

{tools_description}

[Suggestions d'Utilisation des Outils - Veuillez Mélanger Différents Outils, N'Utilisez Pas Un Seul]
- insight_forge : Analyse d'insight approfondie, décompose automatiquement les problèmes et récupère des faits et relations sous plusieurs dimensions
- panorama_search : Recherche panoramique grand angle, comprendre la vue complète des événements, la chronologie et le processus d'évolution
- quick_search : Vérification rapide de points d'information spécifiques
- interview_agents : Interviewer les agents simulés, obtenir des perspectives de première main et des réactions réelles de différents rôles

═══════════════════════════════════════════════════════════════
[Flux de Travail]
═══════════════════════════════════════════════════════════════

Chaque réponse vous permet de faire uniquement l'une des deux choses suivantes (vous ne pouvez pas faire les deux) :

Option A - Appeler un Outil :
Affichez votre réflexion, puis appelez un outil en utilisant le format suivant :
<tool_call&gt;
{{"name": "Nom de l'Outil", "parameters": {{"nom_paramètre": "valeur_paramètre"}}}}
</tool_call&gt;
Le système exécutera l'outil et vous retournera le résultat. Vous n'avez pas besoin et ne pouvez pas écrire vous-même les résultats retournés par les outils.

Option B - Produire le Contenu Final :
Lorsque vous avez recueilli suffisamment d'informations via les outils, commencez par "Final Answer:" et produisez le contenu de la section.

⚠️ Strictement Interdit :
- Interdit d'inclure à la fois des appels d'outils et une réponse finale dans une seule réponse
- Interdit de fabriquer des résultats retournés par les outils (Observation), tous les résultats d'outils sont injectés par le système
- Au maximum un appel d'outil par réponse

═══════════════════════════════════════════════════════════════
[Exigences de Contenu de Section]
═══════════════════════════════════════════════════════════════

1. Le contenu doit être basé sur les données de simulation récupérées par les outils
2. Citez abondamment le texte original pour démontrer les effets de simulation
3. Utilisez le format Markdown (mais interdit d'utiliser des titres) :
   - Utilisez **texte en gras** pour marquer les points clés (remplaçant les sous-titres)
   - Utilisez des listes (- ou 1.2.3.) pour organiser les points
   - Utilisez des lignes vides pour séparer les paragraphes
   - ❌ Interdit d'utiliser toute syntaxe de titre comme #, ##, ###, ####
4. [Spécification du Format de Citation - Doit Être un Paragraphe Séparé]
   Les citations doivent être des paragraphes autonomes avec des lignes vides avant et après, ne peuvent pas être mélangées dans les paragraphes :

   ✅ Format Correct :
   ```
   La réponse des officiels de l'école a été considérée comme manquant de contenu substantiel.

   > « Le mode de réponse de l'école apparaît rigide et lent dans l'environnement des médias sociaux en rapide évolution. »

   Cette évaluation reflète une insatisfaction publique généralisée.
   ```

   ❌ Format Incorrect :
   ```
   La réponse des officiels de l'école a été considérée comme manquant de contenu substantiel.> « Le mode de réponse de l'école... » Cette évaluation reflète...
   ```
5. Maintenez une cohérence logique avec les autres sections
6. [Éviter la Duplication] Lisez attentivement le contenu des sections complétées ci-dessous, ne répétez pas la description des mêmes informations
7. [Rappel Final] N'ajoutez aucun titre ! Utilisez **gras** à la place des sous-titres de section"""

SECTION_USER_PROMPT_TEMPLATE = """\
Contenu des Sections Complétées (Veuillez Lire Attentivement pour Éviter la Duplication) :
{previous_content}

═══════════════════════════════════════════════════════════════
[Tâche Actuelle] Rédiger la Section : {section_title}
═══════════════════════════════════════════════════════════════

[Rappels Importants]
1. Lisez attentivement les sections complétées ci-dessus pour éviter de répéter le même contenu !
2. Vous devez appeler des outils pour obtenir des données de simulation avant de commencer
3. Veuillez mélanger différents outils, n'utilisez pas un seul
4. Le contenu du rapport doit provenir des résultats de récupération, n'utilisez pas vos propres connaissances

[⚠️ Avertissement de Format - À Suivre Impérativement]
- ❌ N'écrivez aucun titre (#, ##, ###, #### aucun autorisé)
- ❌ N'écrivez pas « {section_title} » en ouverture
- ✅ Les titres de section sont ajoutés automatiquement par le système
- ✅ Rédigez le corps directement, utilisez **gras** à la place des sous-titres de section

Veuillez commencer :
1. Réfléchissez d'abord (Réflexion) aux informations dont cette section a besoin
2. Puis appelez des outils (Action) pour obtenir des données de simulation
3. Après avoir recueilli suffisamment d'informations, produisez la réponse finale en commençant par "Final Answer:" (texte pur, sans titres)"""

# ── Modèles de Messages de Boucle ReACT ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (Résultat de Récupération) :

═══ L'outil {tool_name} a renvoyé ═══
{result}

═══════════════════════════════════════════════════════════════
Outils appelés {tool_calls_count}/{max_tool_calls} fois (Utilisés : {used_tools_str}){unused_hint}
- Si les informations sont suffisantes : Commencez par "Final Answer:" et produisez le contenu de la section (vous devez citer le texte original ci-dessus)
- Si plus d'informations sont nécessaires : Appelez un outil pour continuer la récupération
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "[Avis] Vous n'avez appelé que {tool_calls_count} outils, il en faut au moins {min_tool_calls}. "
    'Veuillez appeler à nouveau des outils pour obtenir plus de données de simulation, puis produire la réponse finale (commencez par "Final Answer:"). {unused_hint}'
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Actuellement {tool_calls_count} outils appelés, il en faut au moins {min_tool_calls}. "
    "Veuillez appeler des outils pour obtenir des données de simulation. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Le nombre d'appels d'outils a atteint la limite ({tool_calls_count}/{max_tool_calls}), impossible d'appeler d'autres outils. "
    'Veuillez immédiatement commencer par "Final Answer:" et produire le contenu de la section sur la base des informations acquises.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 Vous n'avez pas encore utilisé : {unused_list}, il est suggéré d'essayer différents outils pour obtenir des informations multi-perspectives"

REACT_FORCE_FINAL_MSG = 'Limite d\'appels d\'outils atteinte, veuillez directement produire "Final Answer:" et générer le contenu de la section.'

# ── Prompt de Chat ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
Vous êtes un assistant de prédiction de simulation concis et efficace.

[Contexte]
Condition de Prédiction : {simulation_requirement}

[Rapport d'Analyse Généré]
{report_content}

[Règles]
1. Prioriser les réponses basées sur le contenu du rapport ci-dessus
2. Répondre directement aux questions, éviter les délibérations longues
3. Appeler des outils pour récupérer plus de données uniquement si le contenu du rapport est insuffisant pour répondre
4. Les réponses doivent être concises, claires et bien organisées

[Outils Disponibles] (utiliser uniquement si nécessaire, appeler au maximum 1-2 fois)
{tools_description}

[Format d'Appel d'Outil]
<tool_call&gt;
{{"name": "Nom de l'Outil", "parameters": {{"nom_paramètre": "valeur_paramètre"}}}}
</tool_call&gt;

[Style de Réponse]
- Concis et direct, ne rédigez pas de passages longs
- Utilisez le format > pour citer le contenu clé
- Donnez d'abord les conclusions, puis expliquez les raisons
- Répondez TOUJOURS en français, quelle que soit la langue utilisée dans le matériel source ou le contenu du rapport"""

CHAT_OBSERVATION_SUFFIX = "\n\nVeuillez répondre à la question de manière concise."


# ═══════════════════════════════════════════════════════════════
# Classe principale ReportAgent
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    ReportAgent - agent de génération de rapport de simulation.

    Utilise le pattern ReACT (raisonnement + action) :
    1. Planification : analyser les besoins et structurer le rapport.
    2. Génération : produire le contenu section par section avec appels d'outils si nécessaire.
    3. Réflexion : vérifier la complétude et la précision.
    """
    
    # Nombre maximal d'appels d'outils par section
    MAX_TOOL_CALLS_PER_SECTION = 5

    # Nombre maximal de tours de réflexion
    MAX_REFLECTION_ROUNDS = 3

    # Nombre maximal d'appels d'outils en conversation
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self,
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        graph_tools: Optional[GraphToolsService] = None
    ):
        """
        Initialiser ReportAgent.

        Args:
            graph_id: ID du graphe.
            simulation_id: ID de simulation.
            simulation_requirement: description du besoin de simulation.
            llm_client: client LLM optionnel.
            graph_tools: service d'outils graphe optionnel, injecté depuis GraphStorage.
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement

        self.llm = llm_client or LLMClient()
        if graph_tools is None:
            raise ValueError(
                "graph_tools (GraphToolsService) est requis. "
                "Créez-le via GraphToolsService(storage=...) et passez-le en paramètre."
            )
        self.graph_tools = graph_tools
        
        # Définitions des outils
        self.tools = self._define_tools()

        # Logger (initialisé dans generate_report)
        self.report_logger: Optional[ReportLogger] = None
        # Logger console (initialisé dans generate_report)
        self.console_logger: Optional[ReportConsoleLogger] = None

        logger.info(f"Initialisation de ReportAgent terminée : graph_id={graph_id}, simulation_id={simulation_id}")
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Définir les outils disponibles"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "La question ou le sujet que vous souhaitez analyser en profondeur",
                    "report_context": "Contexte de la section de rapport actuelle (optionnel, aide à générer des sous-questions plus précises)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Requête de recherche, utilisée pour le tri par pertinence",
                    "include_expired": "Inclure ou non le contenu expiré/historique (par défaut True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Chaîne de requête de recherche",
                    "limit": "Nombre de résultats à retourner (optionnel, par défaut 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Sujet d'entretien ou description des exigences (par ex. 'comprendre les opinions des étudiants sur l'incident de formaldéhyde du dortoir')",
                    "max_agents": "Nombre maximal d'agents à interroger (optionnel, par défaut 5, max 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Exécuter l'appel d'outil

        Args:
            tool_name: Nom de l'outil
            parameters: Paramètres de l'outil
            report_context: Contexte du rapport (pour InsightForge)

        Returns:
            Résultat de l'exécution de l'outil (format texte)
        """
        logger.info(f"Exécution de l'outil : {tool_name}, paramètres : {parameters}")
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.graph_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Recherche panoramique - obtenir une vue complète
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.graph_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Recherche simple - recherche rapide
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.graph_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Entretien approfondi : appeler l'API OASIS réelle pour obtenir les réponses des agents simulés.
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.graph_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Compatibilité Ascendante : Anciens Outils (Redirection Interne vers Nouveaux Outils) ==========

            elif tool_name == "search_graph":
                # Redirigé vers quick_search
                logger.info("search_graph a été redirigé vers quick_search")
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.graph_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.graph_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Redirigé vers insight_forge car il est plus puissant
                logger.info("get_simulation_context a été redirigé vers insight_forge")
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.graph_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Outil inconnu : {tool_name}. Veuillez utiliser l'un des outils suivants : insight_forge, panorama_search, quick_search"

        except Exception as e:
            logger.error(f"Échec de l'exécution de l'outil : {tool_name}, erreur : {str(e)}")
            return f"Échec de l'exécution de l'outil : {str(e)}"
    
    # Ensemble de noms d'outils valides, utilisé pour la validation lors de l'analyse du JSON brut de secours
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Analyser les appels d'outils à partir de la réponse LLM

        Formats pris en charge (par ordre de priorité) :
        1. <tool_call&gt;{"name": "nom_outil", "parameters": {...}}</tool_call&gt;
        2. JSON brut (la réponse entière ou une seule ligne est un JSON d'appel d'outil)
        """
        tool_calls = []

        # Format 1 : Style XML (format standard)
        xml_pattern = r'<tool_call&gt;\s*(\{.*?\})\s*</tool_call&gt;'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2 : Secours - Le LLM produit directement du JSON brut (non enveloppé dans des balises <tool_call&gt;)
        # Essayer uniquement si le format 1 n'a pas correspondu pour éviter les erreurs de correspondance JSON dans le texte
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # La réponse peut contenir du texte de réflexion + JSON brut, essayer d'extraire le dernier objet JSON
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Valider si le JSON analysé est un appel d'outil valide"""
        # Prendre en charge à la fois les noms de clés {"name": ..., "parameters": ...} et {"tool": ..., "params": ...}
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Normaliser les noms de clés en name / parameters
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Générer le texte de description des outils"""
        desc_parts = ["Outils disponibles :"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Paramètres : {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self,
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Planifier le plan du rapport

        Utiliser le LLM pour analyser les exigences de simulation et planifier la structure du rapport

        Args:
            progress_callback: Fonction de rappel de progression

        Returns:
            ReportOutline: Plan du rapport
        """
        logger.info("Début de la planification du plan du rapport...")

        if progress_callback:
            progress_callback("planning", 0, "Analyse des exigences de simulation...")

        # D'abord obtenir le contexte de simulation
        context = self.graph_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )

        if progress_callback:
            progress_callback("planning", 30, "Génération du plan du rapport...")
        
        system_prompt = PLAN_SYSTEM_PROMPT
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, "Analyse de la structure du plan...")

            # Analyser le plan
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Rapport d'analyse de simulation"),
                summary=response.get("summary", ""),
                sections=sections
            )

            if progress_callback:
                progress_callback("planning", 100, "Planification du plan achevée")

            logger.info(f"Planification du plan achevée : {len(sections)} sections")
            return outline

        except Exception as e:
            logger.error(f"Échec de la planification du plan : {str(e)}")
            # Retourner un plan par défaut (3 sections comme secours)
            return ReportOutline(
                title="Rapport de prédiction du futur",
                summary="Tendances futures et analyse des risques basées sur les prédictions de simulation",
                sections=[
                    ReportSection(title="Scénario de prédiction et découvertes principales"),
                    ReportSection(title="Analyse de prédiction du comportement des foules"),
                    ReportSection(title="Perspectives tendancielles et avertissement de risques")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Générer le contenu d'une section individuelle en utilisant le pattern ReACT

        Boucle ReACT :
        1. Réflexion - Analyser de quelles informations on a besoin
        2. Action - Appeler un outil pour obtenir des informations
        3. Observation - Analyser les résultats retournés par l'outil
        4. Répéter jusqu'à ce que les informations soient suffisantes ou que les itérations maximales soient atteintes
        5. Réponse Finale - Générer le contenu de la section

        Args:
            section: Section à générer
            outline: Plan complet
            previous_sections: Contenu des sections précédentes (pour maintenir la cohérence)
            progress_callback: Rappel de progression
            section_index: Index de la section (pour la journalisation)

        Returns:
            Contenu de la section (format Markdown)
        """
        logger.info(f"Génération ReACT de la section : {section.title}")
        
        # Enregistrer le début de la section
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )

        # Construire le prompt utilisateur - passer au maximum 4000 caractères pour chaque section achevée
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Maximum 4000 caractères par section
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(Ceci est la première section)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Boucle ReACT
        tool_calls_count = 0
        max_iterations = 5  # Itérations maximales
        min_tool_calls = 3  # Appels d'outils minimum
        conflict_retries = 0  # Conflits consécutifs où les appels d'outils et la Réponse Finale apparaissent simultanément
        used_tools = set()  # Enregistrer les noms d'outils déjà appelés
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Contexte du rapport pour la génération de sous-questions InsightForge
        report_context = f"Titre de la section : {section.title}\nExigence de simulation : {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    f"Récupération approfondie et rédaction en cours ({tool_calls_count}/{self.MAX_TOOL_CALLS_PER_SECTION})"
                )
            
            # Appeler le LLM
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Vérifier si le retour du LLM est None (exception API ou contenu vide)
            if response is None:
                logger.warning(f"Section {section.title} tour {iteration + 1} : le LLM a retourné None")
                # S'il reste des itérations, ajouter un message et réessayer
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Réponse vide)"})
                    messages.append({"role": "user", "content": "Veuillez continuer à générer le contenu."})
                    continue
                # Dernière itération a aussi retourné None, sortir de la boucle et entrer en conclusion forcée
                break

            logger.debug(f"Réponse LLM : {response[:200]}...")

            # Analyser une fois, réutiliser le résultat
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Gestion des conflits : le LLM produit simultanément des appels d'outils et la Réponse Finale ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    f"Section {section.title} tour {iteration+1} : "
                    f"Le LLM a produit simultanément des appels d'outils et la Réponse Finale ({conflict_retries} conflits)"
                )

                if conflict_retries <= 2:
                    # Les deux premières fois : ignorer cette réponse et demander au LLM de répondre à nouveau
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "[Erreur de format] Vous ne pouvez pas inclure à la fois des appels d'outils et une Réponse Finale dans une seule réponse.\n"
                            "Chaque réponse ne peut faire qu'une seule des choses suivantes :\n"
                            "- Appeler un outil (produire un bloc <tool_call&gt;, ne pas écrire de réponse finale)\n"
                            "- Produire le contenu final (commençant par 'Final Answer:', ne pas inclure <tool_call&gt;)\n"
                            "Veuillez répondre à nouveau et ne faire qu'une seule de ces actions."
                        ),
                    })
                    continue
                else:
                    # Troisième fois : rétrograder, tronquer au premier appel d'outil, forcer l'exécution
                    logger.warning(
                        f"Section {section.title} : {conflict_retries} conflits consécutifs, "
                        "rétrogradé pour tronquer et exécuter le premier appel d'outil"
                    )
                    first_tool_end = response.find('</tool_call&gt;')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call&gt;')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # Enregistrer la réponse du LLM
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Cas 1 : Le LLM produit la Réponse Finale ──
            if has_final_answer:
                # Appels d'outils insuffisants, rejeter et demander de continuer à appeler des outils
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(Ces outils n'ont pas été utilisés, il est recommandé de les utiliser : {', '.join(unused_tools)})" if unused_tools else ""
                    messages.append({
                        "role": "user",
                        "content": REACT_INSUFFICIENT_TOOLS_MSG.format(
                            tool_calls_count=tool_calls_count,
                            min_tool_calls=min_tool_calls,
                            unused_hint=unused_hint,
                        ),
                    })
                    continue

                # Achèvement normal
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(f"Génération de la section {section.title} achevée (appels d'outils : {tool_calls_count} fois)")

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Cas 2 : Le LLM tente d'appeler des outils ──
            if has_tool_calls:
                # Quota d'outils épuisé → informer clairement, demander de produire la Réponse Finale
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Exécuter uniquement le premier appel d'outil
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(f"Le LLM a tenté d'appeler {len(tool_calls)} outils, exécution uniquement du premier : {call['name']}")

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Construire l'indice d'outils inutilisés
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list="、".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Cas 3 : Ni appel d'outil, ni Réponse Finale ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Nombre d'appels d'outils insuffisant, recommander les outils inutilisés
                unused_tools = all_tools - used_tools
                unused_hint = f"(Ces outils n'ont pas été utilisés, il est recommandé de les utiliser : {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Adopter directement ce contenu comme réponse finale, ne plus attendre
            logger.info(f"Section {section.title} : préfixe 'Final Answer:' non détecté, adoption directe de la sortie du LLM comme contenu final (Appels d'outils : {tool_calls_count} fois)")
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Itérations maximales atteintes, forcer la génération du contenu
        logger.warning(f"Section {section.title} : nombre maximal d'itérations atteint, génération forcée")
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Vérifier la conclusion forcée lorsque le LLM retourne None
        if response is None:
            final_answer = f"(La génération de cette section a échoué : le LLM a retourné une réponse vide, veuillez réessayer plus tard)"
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Enregistrer le journal d'achèvement de la génération du contenu de la section
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Générer le rapport complet (sortie en temps réel par section)
        
        Structure des fichiers :
        rapports/{report_id}/
            outline.json    - Plan du rapport
            progress.json   - Progression de la génération
            section_01.md   - Section 1
            section_02.md   - Section 2
            ...
            full_report.md  - Rapport complet
        
        Args:
            report_id : ID du rapport (optionnel, auto-généré si non fourni)
            
        Returns:
            Report: Rapport complet
        """
        import uuid
        
        # Si report_id n'est pas fourni, auto-générer
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # Liste des titres de sections achevées (pour le suivi de la progression)
        completed_section_titles = []
        
        try:
            # Initialisation : Créer le dossier du rapport et sauvegarder l'état initial
            ReportManager._ensure_report_folder(report_id)
            
            # Initialiser le logger structuré (agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Initialiser le logger console (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, "Initialisation du rapport...",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Phase 1 : planifier le plan
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, "Début de la planification du plan du rapport...",
                completed_sections=[]
            )
            
            # Enregistrer le début de la planification du plan
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, "Début de la planification du plan du rapport...")
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Enregistrer le journal d'achèvement de la planification du plan
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Sauvegarder le plan dans un fichier
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, f"Planification du plan achevée, total {len(outline.sections)} sections",
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(f"Plan sauvegardé dans le fichier : {report_id}/outline.json")
            
            # Phase 2 : générer les sections séquentiellement et les enregistrer une par une.
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Conserver le contenu généré pour le contexte.
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Mettre à jour la progression
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    f"Génération de la section : {section.title} ({section_num}/{total_sections})",
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )
                
                if progress_callback:
                    progress_callback(
                        "generating", 
                        base_progress, 
                        f"Génération de la section : {section.title} ({section_num}/{total_sections})"
                    )
                
                # Générer le contenu principal de la section
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Sauvegarder la section
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Enregistrer le journal d'achèvement de la section
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(f"Section sauvegardée : {report_id}/section_{section_num:02d}.md")
                
                # Mettre à jour la progression
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    f"Section {section.title} terminée",
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Phase 3 : assembler le rapport complet
            if progress_callback:
                progress_callback("generating", 95, "Assemblage du rapport complet...")
            
            ReportManager.update_progress(
                report_id, "generating", 95, "Assemblage du rapport complet...",
                completed_sections=completed_section_titles
            )
            
            # Utiliser ReportManager pour assembler le rapport complet
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Calculer le temps total écoulé
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Enregistrer le journal d'achèvement du rapport
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Sauvegarder le rapport final
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, "Génération du rapport terminée",
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, "Génération du rapport terminée")
            
            logger.info(f"Génération du rapport terminée : {report_id}")
            
            # Fermer le logger console
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(f"Échec de la génération du rapport : {str(e)}")
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Enregistrer le journal d'erreur
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Sauvegarder l'état d'échec
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, f"Échec de la génération du rapport : {str(e)}",
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Ignorer l'erreur de sauvegarde d'échec
            
            # Fermer le logger console
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Discussion avec l'Agent de Rapport
        
        En discussion, l'Agent peut appeler de manière autonome des outils de récupération pour répondre aux questions
        
        Args:
            message: Message utilisateur
            chat_history: Historique de discussion
            
        Returns:
            {
                "response": "Réponse de l'Agent",
                "tool_calls": [liste des appels d'outils],
                "sources": [source d'information]
            }
        """
        logger.info(f"Discussion avec l'Agent de Rapport : {message[:50]}...")
        
        chat_history = chat_history or []
        
        # Obtenir le contenu du rapport déjà généré
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Limiter la longueur du rapport, éviter un contexte trop long
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [contenu du rapport tronqué] ..."
        except Exception as e:
            logger.warning(f"Échec de l'obtention du contenu du rapport : {e}")
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(aucun rapport)",
            tools_description=self._get_tools_description(),
        )

        # Construire les messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Ajouter l'historique de discussion
        for h in chat_history[-10:]:  # Limiter la longueur de l'historique
            messages.append(h)
        
        # Ajouter le message utilisateur
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # Boucle ReACT (version simplifiée)
        tool_calls_made = []
        max_iterations = 2  # Réduire les itérations
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Analyser les appels d'outils
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # Pas d'appel d'outil, retourner directement la réponse
                clean_response = re.sub(r'<tool_call&gt;.*?</tool_call&gt;', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Exécuter l'appel d'outil (limiter le nombre)
            tool_results = []
            for call in tool_calls[:1]:  # Au maximum exécuter 1 appel d'outil
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Limiter la longueur du résultat
                })
                tool_calls_made.append(call)
            
            # Ajouter les résultats convertis aux messages
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[Résultat {r['tool']}]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Itération maximale atteinte, obtenir la réponse finale
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Nettoyer la réponse
        clean_response = re.sub(r'<tool_call&gt;.*?</tool_call&gt;', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Gestionnaire de Rapports
    
    Responsable du stockage persistant et de la récupération des rapports
    
    Structure des fichiers (sortie par section) :
    reports/
      {report_id}/
        meta.json          - Méta-informations et statut du rapport
        outline.json       - Plan du rapport
        progress.json      - Progression de la génération
        section_01.md      - Section 1
        section_02.md      - Section 2
        ...
        full_report.md     - Rapport complet
    """
    
    # Répertoire de stockage des rapports
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Garantir l'existence du répertoire racine des rapports"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Obtenir le chemin du dossier du rapport"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Garantir l'existence du dossier du rapport et retourner le chemin"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Obtenir le chemin du fichier de méta-informations du rapport"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Obtenir le chemin du fichier Markdown du rapport complet"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Obtenir le chemin du fichier de plan"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Obtenir le chemin du fichier de progression"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Obtenir le chemin du fichier Markdown de la section"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Obtenir le chemin du fichier de journal de l'Agent"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Obtenir le chemin du fichier de journal console"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Obtenir le contenu du journal console
        
        Il s'agit du journal de sortie console (INFO, WARNING, etc.) pendant le processus de génération du rapport,
        différent du journal structuré agent_log.jsonl.
        
        Args:
            report_id: ID du rapport
            from_line: À partir de quelle ligne commencer la lecture (pour l'obtention incrémentale, 0 signifie depuis le début)
            
        Returns:
            {
                "logs": [liste des lignes de journal],
                "total_lines": nombre total de lignes,
                "from_line": numéro de ligne de départ,
                "has_more": s'il y a plus de journaux
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Conserver la ligne de journal originale, supprimer les caractères de fin de ligne
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Déjà lu jusqu'à la fin
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Obtenir le journal console complet (obtention unique de tout)
        
        Args:
            report_id: ID du rapport
            
        Returns:
            Liste des lignes de journal
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Obtenir le contenu du journal de l'Agent
        
        Args:
            report_id: ID du rapport
            from_line: À partir de quelle ligne commencer la lecture (pour l'obtention incrémentale, 0 signifie depuis le début)
            
        Returns:
            {
                "logs": [liste des entrées de journal],
                "total_lines": nombre total de lignes,
                "from_line": numéro de ligne de départ,
                "has_more": s'il y a plus de journaux
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Ignorer les lignes dont l'analyse a échoué
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Déjà lu jusqu'à la fin
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obtenir le journal complet de l'Agent (pour l'obtention unique de tout)
        
        Args:
            report_id: ID du rapport
            
        Returns:
            Liste des entrées de journal
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Sauvegarder le plan du rapport
        
        Appelé immédiatement après l'achèvement de la phase de planification
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(f"Plan sauvegardé : {report_id}")
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Sauvegarder une section individuelle

        Appelé immédiatement après l'achèvement de la génération de chaque section, implémente la sortie par section

        Args:
            report_id: ID du rapport
            section_index: Index de la section (à partir de 1)
            section: Objet section

        Returns:
            Chemin du fichier sauvegardé
        """
        cls._ensure_report_folder(report_id)

        # Construire le contenu Markdown de la section - nettoyer les titres potentiellement dupliqués
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Sauvegarder le fichier
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(f"Section sauvegardée : {report_id}/{file_suffix}")
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Nettoyer le contenu de la section
        
        1. Supprimer les lignes de titre Markdown en double au début du contenu et du titre de section
        2. Convertir tous les titres de niveau ### et inférieur en texte gras
        
        Args:
            content: Contenu original
            section_title: Titre de la section
            
        Returns:
            Contenu après nettoyage
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Vérifier s'il s'agit d'une ligne de titre Markdown
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Vérifier s'il s'agit d'un titre en double avec le titre de section (ignorer les doublons dans les 5 premières lignes)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Convertir tous les titres de niveau (#, ##, ###, #### etc.) en gras
                # Car le titre de section est ajouté par le système, le contenu ne doit avoir aucun titre
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Ajouter une ligne vide
                continue
            
            # Si la ligne précédente était un titre ignoré, et que la ligne actuelle est vide, l'ignorer aussi
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Supprimer les lignes vides au début
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Supprimer les lignes de séparation au début
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Simultanément supprimer les lignes vides après la ligne de séparation
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Mettre à jour la progression de la génération du rapport
        
        Le frontend peut obtenir la progression en temps réel en lisant progress.json
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Obtenir la progression de la génération du rapport"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Obtenir la liste des sections déjà générées
        
        Retourne les informations de tous les fichiers de sections déjà sauvegardés
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Analyser l'index de la section à partir du nom de fichier
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Assembler le rapport complet
        
        Assembler le rapport complet à partir des fichiers de sections sauvegardés, et nettoyer les titres
        """
        folder = cls._get_report_folder(report_id)
        
        # Construire l'en-tête du rapport
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Lire séquentiellement tous les fichiers de sections
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Post-traitement : nettoyer les problèmes de titres du rapport entier
        md_content = cls._post_process_report(md_content, outline)
        
        # Sauvegarder le rapport complet
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Rapport complet assemblé : {report_id}")
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Post-traiter le contenu du rapport
        
        1. Supprimer les titres en double
        2. Conserver le titre principal du rapport (#) et les titres de section (##), supprimer les titres d'autres niveaux (###, #### etc)
        3. Nettoyer les lignes vides redondantes et les lignes de séparation
        
        Args:
            content: Contenu original du rapport
            outline: Plan du rapport
            
        Returns:
            Contenu après traitement
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Collecter tous les titres de section dans le plan
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Vérifier s'il s'agit d'une ligne de titre
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Vérifier s'il s'agit d'un titre en double (apparaissant dans les 5 lignes consécutives précédentes avec le même contenu de titre)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Ignorer le titre en double et les lignes vides suivantes
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Gestion des niveaux de titre :
                # - # (niveau=1) conserver uniquement le titre principal du rapport
                # - ## (niveau=2) conserver les titres de section
                # - ### et inférieur (niveau>=3) convertir en texte gras
                
                if level == 1:
                    if title == outline.title:
                        # Conserver le titre principal du rapport
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Titre de section utilisant incorrectement #, corriger en ##
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Autre titre de premier niveau convertir en gras
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Conserver le titre de section
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Titre de second niveau non-section convertir en gras
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # Titres de niveau ### et inférieur convertir en texte gras
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Ignorer la ligne de séparation suivant immédiatement un titre
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Après un titre, conserver uniquement une seule ligne vide
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Nettoyer les lignes vides consécutives multiples (conserver au maximum 2)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Sauvegarder les méta-informations et le rapport complet"""
        cls._ensure_report_folder(report.report_id)
        
        # Sauvegarder les méta-informations JSON
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Sauvegarder le plan
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Sauvegarder le rapport Markdown complet
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(f"Rapport sauvegardé : {report.report_id}")
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Obtenir le rapport"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Format compatible antérieur : Vérifier les fichiers stockés directement dans le répertoire reports
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Reconstruire l'objet rapport
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # Si markdown_content est vide, tenter de lire depuis full_report.md
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Obtenir le rapport basé sur l'ID de simulation"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Nouveau format : dossier
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Format compatible antérieur : fichier JSON
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """Lister les rapports"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Nouveau format : dossier
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Format compatible antérieur : fichier JSON
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Tri par date de création décroissante
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Supprimer le rapport (dossier entier)"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # Nouveau format : Supprimer le dossier entier
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(f"Dossier du rapport supprimé : {report_id}")
            return True
        
        # Format compatible antérieur : Supprimer les fichiers individuels
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
