"""
Générateur de profils d'agents OASIS
Convertir les entités du graphe de connaissances au format de profil d'agent requis par la plateforme de simulation OASIS

Améliorations d'optimisation :
1. Appeler la fonction de recherche dans le graphe de connaissances pour enrichir les informations des nœuds
2. Optimiser les prompts pour générer des personas très détaillées
3. Distinguer les entités individuelles des entités de groupe abstraites
"""

import json
import random
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityNode
from ..storage import GraphStorage

logger = get_logger('mirofish.oasis_profile')


@dataclass
class OasisAgentProfile:
    """Structure de données du profil d'agent OASIS"""
    # Champs communs
    user_id: int
    user_name: str
    name: str
    bio: str
    persona: str

    # Champs optionnels - Style Reddit
    karma: int = 1000

    # Champs optionnels - Style Twitter
    friend_count: int = 100
    follower_count: int = 150
    statuses_count: int = 500

    # Informations supplémentaires sur la persona
    age: Optional[int] = None
    gender: Optional[str] = None
    mbti: Optional[str] = None
    country: Optional[str] = None
    profession: Optional[str] = None
    interested_topics: List[str] = field(default_factory=list)

    # Informations sur l'entité source
    source_entity_uuid: Optional[str] = None
    source_entity_type: Optional[str] = None
    
    created_at: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    
    def to_reddit_format(self) -> Dict[str, Any]:
        """Convertir au format de la plateforme Reddit"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # La bibliothèque OASIS requiert le nom de champ username (sans underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "created_at": self.created_at,
        }

        # Ajouter les informations supplémentaires de persona (si disponibles)
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_twitter_format(self) -> Dict[str, Any]:
        """Convertir au format de la plateforme Twitter"""
        profile = {
            "user_id": self.user_id,
            "username": self.user_name,  # La bibliothèque OASIS requiert le nom de champ username (sans underscore)
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "created_at": self.created_at,
        }

        # Ajouter les informations supplémentaires de persona
        if self.age:
            profile["age"] = self.age
        if self.gender:
            profile["gender"] = self.gender
        if self.mbti:
            profile["mbti"] = self.mbti
        if self.country:
            profile["country"] = self.country
        if self.profession:
            profile["profession"] = self.profession
        if self.interested_topics:
            profile["interested_topics"] = self.interested_topics
        
        return profile
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir au format dictionnaire complet"""
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "name": self.name,
            "bio": self.bio,
            "persona": self.persona,
            "karma": self.karma,
            "friend_count": self.friend_count,
            "follower_count": self.follower_count,
            "statuses_count": self.statuses_count,
            "age": self.age,
            "gender": self.gender,
            "mbti": self.mbti,
            "country": self.country,
            "profession": self.profession,
            "interested_topics": self.interested_topics,
            "source_entity_uuid": self.source_entity_uuid,
            "source_entity_type": self.source_entity_type,
            "created_at": self.created_at,
        }


class OasisProfileGenerator:
    """
    Générateur de profils OASIS

    Convertir les entités du graphe de connaissances en profils d'agents requis par la simulation OASIS

    Fonctionnalités d'optimisation :
    1. Appeler la fonction de recherche dans le graphe de connaissances pour obtenir un contexte plus riche
    2. Générer des personas très détaillées (y compris informations de base, expérience professionnelle, traits de personnalité, comportement sur les médias sociaux, etc.)
    3. Distinguer les entités individuelles des entités de groupe abstraites
    """

    # Liste des types MBTI
    MBTI_TYPES = [
        "INTJ", "INTP", "ENTJ", "ENTP",
        "INFJ", "INFP", "ENFJ", "ENFP",
        "ISTJ", "ISFJ", "ESTJ", "ESFJ",
        "ISTP", "ISFP", "ESTP", "ESFP"
    ]

    # Liste des pays courants
    COUNTRIES = [
        "US", "UK", "Japan", "Germany", "France",
        "Canada", "Australia", "Brazil", "India", "South Korea"
    ]

    # Entités de type individuel (nécessitent la génération de personas spécifiques)
    INDIVIDUAL_ENTITY_TYPES = [
        "student", "alumni", "professor", "person", "publicfigure",
        "expert", "faculty", "official", "journalist", "activist"
    ]

    # Entités de type groupe/institutionnel (nécessitent la génération de personas de représentants de groupe)
    GROUP_ENTITY_TYPES = [
        "university", "governmentagency", "organization", "ngo",
        "mediaoutlet", "company", "institution", "group", "community"
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        storage: Optional[GraphStorage] = None,
        graph_id: Optional[str] = None
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

        # GraphStorage pour l'enrichissement par recherche hybride
        self.storage = storage
        self.graph_id = graph_id
    
    def generate_profile_from_entity(
        self,
        entity: EntityNode,
        user_id: int,
        use_llm: bool = True
    ) -> OasisAgentProfile:
        """
        Générer le profil d'agent OASIS à partir de l'entité du graphe de connaissances

        Args:
            entity: Nœud d'entité du graphe de connaissances
            user_id: ID utilisateur (pour OASIS)
            use_llm: Indique s'il faut utiliser le LLM pour générer une persona détaillée

        Returns:
            OasisAgentProfile
        """
        entity_type = entity.get_entity_type() or "Entity"

        # Informations de base
        name = entity.name
        user_name = self._generate_username(name)

        # Construire les informations contextuelles
        context = self._build_entity_context(entity)
        
        if use_llm:
            # Utiliser le LLM pour générer une persona détaillée
            profile_data = self._generate_profile_with_llm(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes,
                context=context
            )
        else:
            # Utiliser des règles pour générer une persona basique
            profile_data = self._generate_profile_rule_based(
                entity_name=name,
                entity_type=entity_type,
                entity_summary=entity.summary,
                entity_attributes=entity.attributes
            )
        
        return OasisAgentProfile(
            user_id=user_id,
            user_name=user_name,
            name=name,
            bio=profile_data.get("bio", f"{entity_type}: {name}"),
            persona=profile_data.get("persona", entity.summary or f"Un(e) {entity_type} nommé(e) {name}."),
            karma=profile_data.get("karma", random.randint(500, 5000)),
            friend_count=profile_data.get("friend_count", random.randint(50, 500)),
            follower_count=profile_data.get("follower_count", random.randint(100, 1000)),
            statuses_count=profile_data.get("statuses_count", random.randint(100, 2000)),
            age=profile_data.get("age"),
            gender=profile_data.get("gender"),
            mbti=profile_data.get("mbti"),
            country=profile_data.get("country"),
            profession=profile_data.get("profession"),
            interested_topics=profile_data.get("interested_topics", []),
            source_entity_uuid=entity.uuid,
            source_entity_type=entity_type,
        )
    
    def _generate_username(self, name: str) -> str:
        """Générer le nom d'utilisateur"""
        # Supprimer les caractères spéciaux, convertir en minuscules
        username = name.lower().replace(" ", "_")
        username = ''.join(c for c in username if c.isalnum() or c == '_')

        # Ajouter un suffixe aléatoire pour éviter les doublons
        suffix = random.randint(100, 999)
        return f"{username}_{suffix}"
    
    def _search_graph_for_entity(self, entity: EntityNode) -> Dict[str, Any]:
        """
        Utiliser la recherche hybride GraphStorage pour obtenir des informations riches liées à l'entité

        Utilise storage.search() (vectoriel hybride + BM25) pour les arêtes et les nœuds.

        Args:
            entity: Objet nœud d'entité

        Returns:
            Dictionnaire contenant facts, node_summaries, context
        """
        if not self.storage:
            return {"facts": [], "node_summaries": [], "context": ""}

        entity_name = entity.name

        results = {
            "facts": [],
            "node_summaries": [],
            "context": ""
        }

        if not self.graph_id:
            logger.debug(f"Recherche dans le graphe de connaissances ignorée : graph_id non défini")
            return results

        comprehensive_query = f"Toutes les informations, activités, événements, relations et antécédents sur {entity_name}"

        try:
            # Rechercher les arêtes (faits)
            edge_results = self.storage.search(
                graph_id=self.graph_id,
                query=comprehensive_query,
                limit=30,
                scope="edges"
            )

            all_facts = set()
            if isinstance(edge_results, dict) and 'edges' in edge_results:
                for edge in edge_results['edges']:
                    fact = edge.get('fact', '')
                    if fact:
                        all_facts.add(fact)
            results["facts"] = list(all_facts)

            # Rechercher les nœuds (résumés d'entités)
            node_results = self.storage.search(
                graph_id=self.graph_id,
                query=comprehensive_query,
                limit=20,
                scope="nodes"
            )

            all_summaries = set()
            if isinstance(node_results, dict) and 'nodes' in node_results:
                for node in node_results['nodes']:
                    summary = node.get('summary', '')
                    if summary:
                        all_summaries.add(summary)
                    name = node.get('name', '')
                    if name and name != entity_name:
                        all_summaries.add(f"Entité connexe : {name}")
            results["node_summaries"] = list(all_summaries)

            # Construire le contexte combiné
            context_parts = []
            if results["facts"]:
                context_parts.append("Informations factuelles :\n" + "\n".join(f"- {f}" for f in results["facts"][:20]))
            if results["node_summaries"]:
                context_parts.append("Entités connexes :\n" + "\n".join(f"- {s}" for s in results["node_summaries"][:10]))
            results["context"] = "\n\n".join(context_parts)

            logger.info(f"Recherche hybride dans le graphe de connaissances terminée : {entity_name}, {len(results['facts'])} faits récupérés, {len(results['node_summaries'])} nœuds connexes")

        except Exception as e:
            logger.warning(f"Échec de la recherche dans le graphe de connaissances ({entity_name}) : {e}")

        return results
    
    def _build_entity_context(self, entity: EntityNode) -> str:
        """
        Construire les informations contextuelles complètes pour l'entité

        Inclut :
        1. Les informations d'arêtes de l'entité elle-même (faits)
        2. Les informations détaillées des nœuds associés
        3. Les informations riches récupérées par la recherche hybride dans le graphe de connaissances
        """
        context_parts = []

        # 1. Ajouter les informations d'attributs de l'entité
        if entity.attributes:
            attrs = []
            for key, value in entity.attributes.items():
                if value and str(value).strip():
                    attrs.append(f"- {key}: {value}")
            if attrs:
                context_parts.append("### Attributs de l'entité\n" + "\n".join(attrs))

        # 2. Ajouter les informations d'arêtes connexes (faits/relations)
        existing_facts = set()
        if entity.related_edges:
            relationships = []
            for edge in entity.related_edges:  # Pas de limite de quantité
                fact = edge.get("fact", "")
                edge_name = edge.get("edge_name", "")
                direction = edge.get("direction", "")

                if fact:
                    relationships.append(f"- {fact}")
                    existing_facts.add(fact)
                elif edge_name:
                    if direction == "outgoing":
                        relationships.append(f"- {entity.name} --[{edge_name}]--> (Entité connexe)")
                    else:
                        relationships.append(f"- (Entité connexe) --[{edge_name}]--> {entity.name}")

            if relationships:
                context_parts.append("### Faits et relations connexes\n" + "\n".join(relationships))

        # 3. Ajouter les informations détaillées des nœuds connexes
        if entity.related_nodes:
            related_info = []
            for node in entity.related_nodes:  # Pas de limite de quantité
                node_name = node.get("name", "")
                node_labels = node.get("labels", [])
                node_summary = node.get("summary", "")

                # Filtrer les labels par défaut
                custom_labels = [l for l in node_labels if l not in ["Entity", "Node"]]
                label_str = f" ({', '.join(custom_labels)})" if custom_labels else ""

                if node_summary:
                    related_info.append(f"- **{node_name}**{label_str}: {node_summary}")
                else:
                    related_info.append(f"- **{node_name}**{label_str}")

            if related_info:
                context_parts.append("### Informations sur les entités connexes\n" + "\n".join(related_info))

        # 4. Utiliser la recherche hybride dans le graphe de connaissances pour obtenir des informations plus riches
        graph_results = self._search_graph_for_entity(entity)

        if graph_results.get("facts"):
            # Déduplication : exclure les faits existants
            new_facts = [f for f in graph_results["facts"] if f not in existing_facts]
            if new_facts:
                context_parts.append("### Faits récupérés du graphe de connaissances\n" + "\n".join(f"- {f}" for f in new_facts[:15]))

        if graph_results.get("node_summaries"):
            context_parts.append("### Nœuds connexes récupérés du graphe de connaissances\n" + "\n".join(f"- {s}" for s in graph_results["node_summaries"][:10]))
        
        return "\n\n".join(context_parts)
    
    def _is_individual_entity(self, entity_type: str) -> bool:
        """Déterminer si l'entité est de type individuel"""
        return entity_type.lower() in self.INDIVIDUAL_ENTITY_TYPES

    def _is_group_entity(self, entity_type: str) -> bool:
        """Déterminer si l'entité est de type groupe/institutionnel"""
        return entity_type.lower() in self.GROUP_ENTITY_TYPES
    
    def _generate_profile_with_llm(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> Dict[str, Any]:
        """
        Utiliser le LLM pour générer une persona très détaillée

        En fonction du type d'entité :
        - Entités individuelles : générer des profils de personnages spécifiques
        - Entités de groupe/institutionnelles : générer des profils de comptes représentatifs
        """

        is_individual = self._is_individual_entity(entity_type)

        if is_individual:
            prompt = self._build_individual_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )
        else:
            prompt = self._build_group_persona_prompt(
                entity_name, entity_type, entity_summary, entity_attributes, context
            )

        # Essayer plusieurs fois jusqu'à succès ou nombre maximal de tentatives
        max_attempts = 3
        last_error = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self._get_system_prompt(is_individual)},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # Baisser la température à chaque tentative
                    # Ne pas définir max_tokens, laisser le LLM générer librement
                )

                content = response.choices[0].message.content

                # Vérifier si la sortie a été tronquée (finish_reason n'est pas 'stop')
                finish_reason = response.choices[0].finish_reason
                if finish_reason == 'length':
                    logger.warning(f"Sortie du LLM tronquée (tentative {attempt+1}), tentative de réparation...")
                    content = self._fix_truncated_json(content)

                # Essayer d'analyser le JSON
                try:
                    result = json.loads(content)

                    # Valider les champs requis
                    if "bio" not in result or not result["bio"]:
                        result["bio"] = entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}"
                    if "persona" not in result or not result["persona"]:
                        result["persona"] = entity_summary or f"{entity_name} est un(e) {entity_type}."

                    return result

                except json.JSONDecodeError as je:
                    logger.warning(f"Échec de l'analyse JSON (tentative {attempt+1}) : {str(je)[:80]}")

                    # Essayer de réparer le JSON
                    result = self._try_fix_json(content, entity_name, entity_type, entity_summary)
                    if result.get("_fixed"):
                        del result["_fixed"]
                        return result

                    last_error = je

            except Exception as e:
                logger.warning(f"Échec de l'appel LLM (tentative {attempt+1}) : {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(1 * (attempt + 1))  # Temporisation exponentielle

        logger.warning(f"Échec de la génération de persona par LLM ({max_attempts} tentatives) : {last_error}, utilisation de la génération basée sur des règles")
        return self._generate_profile_rule_based(
            entity_name, entity_type, entity_summary, entity_attributes
        )
    
    def _fix_truncated_json(self, content: str) -> str:
        """Réparer le JSON tronqué (sortie tronquée par la limite max_tokens)"""
        import re

        # Si le JSON est tronqué, essayer de le fermer
        content = content.strip()

        # Compter les parenthèses non fermées
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')

        # Vérifier les chaînes non fermées
        # Vérification simple : si le dernier caractère n'est pas une virgule ou un crochet fermant, la chaîne est peut-être tronquée
        if content and content[-1] not in '",}]':
            # Essayer de fermer la chaîne
            content += '"'

        # Fermer les parenthèses
        content += ']' * open_brackets
        content += '}' * open_braces

        return content
    
    def _try_fix_json(self, content: str, entity_name: str, entity_type: str, entity_summary: str = "") -> Dict[str, Any]:
        """Essayer de réparer le JSON corrompu"""
        import re

        # 1. D'abord essayer de réparer le cas tronqué
        content = self._fix_truncated_json(content)

        # 2. Essayer d'extraire la portion JSON
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()

            # 3. Gérer les problèmes de sauts de ligne dans les chaînes
            # Trouver toutes les valeurs de chaîne et remplacer les sauts de ligne
            def fix_string_newlines(match):
                s = match.group(0)
                # Remplacer les sauts de ligne réels dans la chaîne par des espaces
                s = s.replace('\n', ' ').replace('\r', ' ')
                # Remplacer les espaces excédentaires
                s = re.sub(r'\s+', ' ', s)
                return s

            # Correspondre aux valeurs de chaîne JSON
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string_newlines, json_str)

            # 4. Essayer d'analyser
            try:
                result = json.loads(json_str)
                result["_fixed"] = True
                return result
            except json.JSONDecodeError as e:
                # 5. Si toujours échoué, essayer une réparation plus agressive
                try:
                    # Supprimer tous les caractères de contrôle
                    json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                    # Remplacer tous les espaces consécutifs
                    json_str = re.sub(r'\s+', ' ', json_str)
                    result = json.loads(json_str)
                    result["_fixed"] = True
                    return result
                except:
                    pass

        # 6. Essayer d'extraire des informations partielles du contenu
        bio_match = re.search(r'"bio"\s*:\s*"([^"]*)"', content)
        persona_match = re.search(r'"persona"\s*:\s*"([^"]*)', content)  # Peut être tronqué

        bio = bio_match.group(1) if bio_match else (entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}")
        persona = persona_match.group(1) if persona_match else (entity_summary or f"{entity_name} est un(e) {entity_type}.")

        # Si du contenu significatif a été extrait, marquer comme réparé
        if bio_match or persona_match:
            logger.info(f"Informations partielles extraites du JSON corrompu")
            return {
                "bio": bio,
                "persona": persona,
                "_fixed": True
            }

        # 7. Échec complet, retourner la structure de base
        logger.warning(f"Échec de la réparation JSON, retour de la structure de base")
        return {
            "bio": entity_summary[:200] if entity_summary else f"{entity_type}: {entity_name}",
            "persona": entity_summary or f"{entity_name} est un(e) {entity_type}."
        }
    
    def _get_system_prompt(self, is_individual: bool) -> str:
        """Obtenir le prompt système"""
        base_prompt = "Vous êtes un expert en génération de profils d'utilisateurs de médias sociaux. Générez des personas détaillées et réalistes pour la simulation d'opinions maximisant la restauration de la réalité existante. Doit retourner un format JSON valide avec toutes les valeurs de chaîne ne contenant pas de sauts de ligne non échappés. Utilisez l'anglais."
        return base_prompt
    
    def _build_individual_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Construire le prompt de persona détaillée pour les entités individuelles"""

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "Aucun"
        context_str = context[:3000] if context else "Aucun contexte additionnel"

        return f"""Générez une persona d'utilisateur de médias sociaux détaillée pour l'entité, maximisant la restauration de la réalité existante.

Nom de l'entité : {entity_name}
Type d'entité : {entity_type}
Résumé de l'entité : {entity_summary}
Attributs de l'entité : {attrs_str}

Informations contextuelles :
{context_str}

Veuillez générer un JSON contenant les champs suivants :

1. bio : Biographie de médias sociaux, 200 caractères
2. persona : Description détaillée de la persona (2000 mots de texte pur), doit inclure :
   - Informations de base (âge, profession, formation, lieu de résidence)
   - Antécédents personnels (expériences importantes, associations avec des événements, relations sociales)
   - Traits de personnalité (type MBTI, personnalité centrale, expression émotionnelle)
   - Comportement sur les médias sociaux (fréquence de publication, préférences de contenu, style d'interaction, caractéristiques linguistiques)
   - Positions et opinions (attitudes envers les sujets, contenu pouvant provoquer/toucher les émotions)
   - Caractéristiques uniques (expressions fétiches, expériences spéciales, centres d'intérêt personnels)
   - Souvenirs personnels (partie importante de la persona, présenter l'association de cet individu avec les événements et ses actions/réactions existantes dans les événements)
3. age : Âge en tant que nombre (doit être un entier)
4. gender : Genre, doit être en anglais : "male" ou "female"
5. mbti : Type MBTI (par ex., INTJ, ENFP)
6. country : Pays (utiliser l'anglais, par ex., "US")
7. profession : Profession
8. interested_topics : Tableau de sujets d'intérêt

Important :
- Toutes les valeurs de champ doivent être des chaînes ou des nombres, ne pas utiliser de sauts de ligne
- persona doit être une description textuelle cohérente
- Utiliser l'anglais
- Le contenu doit être cohérent avec les informations de l'entité
- age doit être un entier valide, gender doit être "male" ou "female"
"""

    def _build_group_persona_prompt(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any],
        context: str
    ) -> str:
        """Construire le prompt de persona détaillée pour les entités de groupe/institutionnelles"""

        attrs_str = json.dumps(entity_attributes, ensure_ascii=False) if entity_attributes else "Aucun"
        context_str = context[:3000] if context else "Aucun contexte additionnel"

        return f"""Générez un profil de compte de médias sociaux détaillé pour l'entité institutionnelle/de groupe, maximisant la restauration de la réalité existante.

Nom de l'entité : {entity_name}
Type d'entité : {entity_type}
Résumé de l'entité : {entity_summary}
Attributs de l'entité : {attrs_str}

Informations contextuelles :
{context_str}

Veuillez générer un JSON contenant les champs suivants :

1. bio : Biographie du compte officiel, 200 caractères, professionnelle et appropriée
2. persona : Description détaillée du profil de compte (2000 mots de texte pur), doit inclure :
   - Informations institutionnelles de base (nom officiel, nature de l'organisation, contexte de fondation, fonctions principales)
   - Positionnement du compte (type de compte, audience cible, fonctions principales)
   - Style de communication (caractéristiques linguistiques, expressions courantes, sujets tabous)
   - Caractéristiques de publication de contenu (types de contenu, fréquence de publication, périodes d'activité)
   - Position et attitude (position officielle sur les sujets centraux, gestion des controverses)
   - Notes spéciales (profils de groupe représentés, habitudes opérationnelles)
   - Souvenirs institutionnels (partie importante de la persona institutionnelle, présenter l'association de cette institution avec les événements et ses actions/réactions existantes dans les événements)
3. age : Fixé à 30 (âge virtuel du compte institutionnel)
4. gender : Fixé à "other" (le compte institutionnel utilise other pour désigner un non-individu)
5. mbti : Type MBTI utilisé pour décrire le style du compte, par ex., ISTJ représente un conservateur rigoureux
6. country : Pays (utiliser l'anglais, par ex., "US")
7. profession : Description de la fonction institutionnelle
8. interested_topics : Tableau de domaines d'intérêt

Important :
- Toutes les valeurs de champ doivent être des chaînes ou des nombres, aucune valeur null autorisée
- persona doit être une description textuelle cohérente, n'utilisez pas de sauts de ligne
- Utiliser l'anglais
- age doit être l'entier 30, gender doit être la chaîne "other"
- Le discours du compte institutionnel doit correspondre à son positionnement d'identité"""
    
    def _generate_profile_rule_based(
        self,
        entity_name: str,
        entity_type: str,
        entity_summary: str,
        entity_attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Générer une persona basique en utilisant des règles"""

        # Générer différentes personas en fonction du type d'entité
        entity_type_lower = entity_type.lower()

        if entity_type_lower in ["student", "alumni"]:
            return {
                "bio": f"{entity_type} avec des intérêts dans les études et les questions sociales.",
                "persona": f"{entity_name} est un(e) {entity_type.lower()} qui participe activement aux discussions académiques et sociales. Il/Elle aime partager ses perspectives et se connecter avec ses pairs.",
                "age": random.randint(18, 30),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": "Étudiant",
                "interested_topics": ["Éducation", "Questions sociales", "Technologie"],
            }

        elif entity_type_lower in ["publicfigure", "expert", "faculty"]:
            return {
                "bio": f"Expert et leader d'opinion dans son domaine.",
                "persona": f"{entity_name} est un(e) {entity_type.lower()} reconnu(e) qui partage des analyses et des opinions sur des questions importantes. Il/Elle est connu(e) pour son expertise et son influence dans le débat public.",
                "age": random.randint(35, 60),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(["ENTJ", "INTJ", "ENTP", "INTP"]),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_attributes.get("occupation", "Expert"),
                "interested_topics": ["Politique", "Économie", "Culture et société"],
            }

        elif entity_type_lower in ["mediaoutlet", "socialmediaplatform"]:
            return {
                "bio": f"Compte officiel de {entity_name}. Actualités et mises à jour.",
                "persona": f"{entity_name} est une entité médiatique qui rapporte les actualités et facilite le débat public. Le compte partage des mises à jour en temps opportun et interagit avec l'audience sur l'actualité.",
                "age": 30,  # Âge virtuel institutionnel
                "gender": "other",  # Institutionnel utilise other
                "mbti": "ISTJ",  # Style institutionnel : conservateur rigoureux
                "country": "US",
                "profession": "Média",
                "interested_topics": ["Actualités générales", "Événements en cours", "Affaires publiques"],
            }

        elif entity_type_lower in ["university", "governmentagency", "ngo", "organization"]:
            return {
                "bio": f"Compte officiel de {entity_name}.",
                "persona": f"{entity_name} est une entité institutionnelle qui communique des positions officielles, des annonces et interagit avec les parties prenantes sur les questions pertinentes.",
                "age": 30,  # Âge virtuel institutionnel
                "gender": "other",  # Institutionnel utilise other
                "mbti": "ISTJ",  # Style institutionnel : conservateur rigoureux
                "country": "US",
                "profession": entity_type,
                "interested_topics": ["Politique publique", "Communauté", "Annonces officielles"],
            }

        else:
            # Persona par défaut
            return {
                "bio": entity_summary[:150] if entity_summary else f"{entity_type}: {entity_name}",
                "persona": entity_summary or f"{entity_name} est un(e) {entity_type.lower()} participant(e) aux discussions sociales.",
                "age": random.randint(25, 50),
                "gender": random.choice(["male", "female"]),
                "mbti": random.choice(self.MBTI_TYPES),
                "country": random.choice(self.COUNTRIES),
                "profession": entity_type,
                "interested_topics": ["Général", "Questions sociales"],
            }
    
    def set_graph_id(self, graph_id: str):
        """Définir l'ID du graphe de connaissances pour la recherche"""
        self.graph_id = graph_id
    
    def generate_profiles_from_entities(
        self,
        entities: List[EntityNode],
        use_llm: bool = True,
        progress_callback: Optional[callable] = None,
        graph_id: Optional[str] = None,
        parallel_count: int = 5,
        realtime_output_path: Optional[str] = None,
        output_platform: str = "reddit"
    ) -> List[OasisAgentProfile]:
        """
        Générer les profils d'agents par lots à partir des entités (prend en charge la génération parallèle)

        Args:
            entities: Liste d'entités
            use_llm: Indique s'il faut utiliser le LLM pour générer des personas détaillées
            progress_callback: Fonction de rappel de progression (actuel, total, message)
            graph_id: ID du graphe de connaissances pour la recherche afin d'obtenir un contexte plus riche
            parallel_count: Nombre de générations parallèles, par défaut 5
            realtime_output_path: Chemin du fichier de sortie en temps réel (si fourni, écrit après chaque génération)
            output_platform: Format de plateforme de sortie ("reddit" ou "twitter")

        Returns:
            Liste de profils d'agents
        """
        import concurrent.futures
        from threading import Lock
        
        # Définir graph_id pour la recherche dans le graphe de connaissances
        if graph_id:
            self.graph_id = graph_id

        total = len(entities)
        profiles = [None] * total  # Pré-allouer la liste pour maintenir l'ordre
        completed_count = [0]  # Utiliser une liste pour la modification dans la fermeture
        lock = Lock()

        # Fonction auxiliaire pour l'écriture en temps réel dans le fichier
        def save_profiles_realtime():
            """Sauvegarde en temps réel des profils générés dans le fichier"""
            if not realtime_output_path:
                return

            with lock:
                # Filtrer les profils générés
                existing_profiles = [p for p in profiles if p is not None]
                if not existing_profiles:
                    return

                try:
                    if output_platform == "reddit":
                        # Format JSON Reddit
                        profiles_data = [p.to_reddit_format() for p in existing_profiles]
                        with open(realtime_output_path, 'w', encoding='utf-8') as f:
                            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
                    else:
                        # Format CSV Twitter
                        import csv
                        profiles_data = [p.to_twitter_format() for p in existing_profiles]
                        if profiles_data:
                            fieldnames = list(profiles_data[0].keys())
                            with open(realtime_output_path, 'w', encoding='utf-8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                writer.writeheader()
                                writer.writerows(profiles_data)
                except Exception as e:
                    logger.warning(f"Échec de la sauvegarde en temps réel du profil : {e}")
        
        def generate_single_profile(idx: int, entity: EntityNode) -> tuple:
            """Fonction de travail pour générer un profil unique"""
            entity_type = entity.get_entity_type() or "Entity"

            try:
                profile = self.generate_profile_from_entity(
                    entity=entity,
                    user_id=idx,
                    use_llm=use_llm
                )

                # Sortie en temps réel de la persona générée vers la console et le journal
                self._print_generated_profile(entity.name, entity_type, profile)

                return idx, profile, None

            except Exception as e:
                logger.error(f"Échec de la génération de persona pour l'entité {entity.name} : {str(e)}")
                # Créer un profil de secours
                fallback_profile = OasisAgentProfile(
                    user_id=idx,
                    user_name=self._generate_username(entity.name),
                    name=entity.name,
                    bio=f"{entity_type}: {entity.name}",
                    persona=entity.summary or f"Un(e) participant(e) aux discussions sociales.",
                    source_entity_uuid=entity.uuid,
                    source_entity_type=entity_type,
                )
                return idx, fallback_profile, str(e)

        logger.info(f"Début de la génération parallèle de {total} personas d'agents (nombre parallèle : {parallel_count})...")
        print(f"\n{'='*60}")
        print(f"Début de la génération de personas d'agents - {total} entités au total, nombre parallèle : {parallel_count}")
        print(f"{'='*60}\n")
        
        # Utiliser le pool de threads pour l'exécution parallèle
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_count) as executor:
            # Soumettre toutes les tâches
            future_to_entity = {
                executor.submit(generate_single_profile, idx, entity): (idx, entity)
                for idx, entity in enumerate(entities)
            }

            # Collecter les résultats
            for future in concurrent.futures.as_completed(future_to_entity):
                idx, entity = future_to_entity[future]
                entity_type = entity.get_entity_type() or "Entity"

                try:
                    result_idx, profile, error = future.result()
                    profiles[result_idx] = profile

                    with lock:
                        completed_count[0] += 1
                        current = completed_count[0]

                    # Écriture en temps réel dans le fichier
                    save_profiles_realtime()

                    if progress_callback:
                        progress_callback(
                            current,
                            total,
                            f"Terminé {current}/{total} : {entity.name} ({entity_type})"
                        )

                    if error:
                        logger.warning(f"[{current}/{total}] {entity.name} utilise la persona de secours : {error}")
                    else:
                        logger.info(f"[{current}/{total}] Persona générée avec succès : {entity.name} ({entity_type})")

                except Exception as e:
                    logger.error(f"Une exception s'est produite lors du traitement de l'entité {entity.name} : {str(e)}")
                    with lock:
                        completed_count[0] += 1
                    profiles[idx] = OasisAgentProfile(
                        user_id=idx,
                        user_name=self._generate_username(entity.name),
                        name=entity.name,
                        bio=f"{entity_type}: {entity.name}",
                        persona=entity.summary or "Un(e) participant(e) aux discussions sociales.",
                        source_entity_uuid=entity.uuid,
                        source_entity_type=entity_type,
                    )
                    # Écriture en temps réel dans le fichier (même pour les personas de secours)
                    save_profiles_realtime()

        print(f"\n{'='*60}")
        print(f"Génération de personas terminée ! {len([p for p in profiles if p])} agents générés")
        print(f"{'='*60}\n")
        
        return profiles
    
    def _print_generated_profile(self, entity_name: str, entity_type: str, profile: OasisAgentProfile):
        """Sortie en temps réel de la persona générée vers la console (contenu complet, non tronqué)"""
        separator = "-" * 70

        # Construire le contenu de sortie complet (non tronqué)
        topics_str = ', '.join(profile.interested_topics) if profile.interested_topics else 'Aucun'

        output_lines = [
            f"\n{separator}",
            f"[Généré] {entity_name} ({entity_type})",
            f"{separator}",
            f"Nom d'utilisateur : {profile.user_name}",
            f"",
            f"[Biographie]",
            f"{profile.bio}",
            f"",
            f"[Persona détaillée]",
            f"{profile.persona}",
            f"",
            f"[Attributs de base]",
            f"Âge : {profile.age} | Genre : {profile.gender} | MBTI : {profile.mbti}",
            f"Profession : {profile.profession} | Pays : {profile.country}",
            f"Sujets d'intérêt : {topics_str}",
            separator
        ]

        output = "\n".join(output_lines)

        # Sortie uniquement vers la console (éviter la duplication, le journal ne sort plus le contenu complet)
        print(output)
    
    def save_profiles(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """
        Sauvegarder les profils dans un fichier (choisir le format correct en fonction de la plateforme)

        Exigences de format de la plateforme OASIS :
        - Twitter : format CSV
        - Reddit : format JSON

        Args:
            profiles: Liste de profils
            file_path: Chemin du fichier
            platform: Type de plateforme ("reddit" ou "twitter")
        """
        if platform == "twitter":
            self._save_twitter_csv(profiles, file_path)
        else:
            self._save_reddit_json(profiles, file_path)
    
    def _save_twitter_csv(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Sauvegarder le profil Twitter au format CSV (conforme aux exigences officielles OASIS)

        Champs CSV requis par OASIS Twitter :
        - user_id : ID utilisateur (commençant à 0 selon l'ordre CSV)
        - name : Nom réel de l'utilisateur
        - username : Nom d'utilisateur dans le système
        - user_char : Description détaillée de la persona (injectée dans le prompt système du LLM, guide le comportement de l'agent)
        - description : Biographie publique courte (affichée sur la page de profil utilisateur)

        Différence entre user_char et description :
        - user_char : Usage interne, prompt système du LLM, détermine comment l'agent pense et agit
        - description : Affichage externe, visible par les autres utilisateurs
        """
        import csv

        # S'assurer que l'extension du fichier est .csv
        if not file_path.endswith('.csv'):
            file_path = file_path.replace('.json', '.csv')

        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Écrire l'en-tête requis par OASIS
            headers = ['user_id', 'name', 'username', 'user_char', 'description']
            writer.writerow(headers)

            # Écrire les lignes de données
            for idx, profile in enumerate(profiles):
                # user_char : Persona complète (bio + persona) pour le prompt système du LLM
                user_char = profile.bio
                if profile.persona and profile.persona != profile.bio:
                    user_char = f"{profile.bio} {profile.persona}"
                # Gérer les sauts de ligne (remplacer par un espace dans le CSV)
                user_char = user_char.replace('\n', ' ').replace('\r', ' ')

                # description : Biographie courte pour l'affichage externe
                description = profile.bio.replace('\n', ' ').replace('\r', ' ')

                row = [
                    idx,                    # user_id : ID séquentiel commençant à 0
                    profile.name,           # name : Nom réel
                    profile.user_name,      # username : Nom d'utilisateur
                    user_char,              # user_char : Persona complète (usage interne LLM)
                    description             # description : Bio courte (affichage externe)
                ]
                writer.writerow(row)

        logger.info(f"Sauvegardé {len(profiles)} profils Twitter dans {file_path} (format CSV OASIS)")
    
    def _normalize_gender(self, gender: Optional[str]) -> str:
        """
        Normaliser le champ genre au format anglais requis par OASIS

        OASIS requiert : male, female, other
        """
        if not gender:
            return "other"

        gender_lower = gender.lower().strip()

        # Mappage des genres
        gender_map = {
            "male": "male",
            "female": "female",
            "other": "other",
        }

        return gender_map.get(gender_lower, "other")
    
    def _save_reddit_json(self, profiles: List[OasisAgentProfile], file_path: str):
        """
        Sauvegarder le profil Reddit au format JSON

        Utiliser un format cohérent avec to_reddit_format() pour s'assurer qu'OASIS peut lire correctement.
        Doit inclure le champ user_id, qui est la clé pour la correspondance OASIS agent_graph.get_agent() !

        Champs requis :
        - user_id : ID utilisateur (entier, utilisé pour correspondre à poster_agent_id dans initial_posts)
        - username : Nom d'utilisateur
        - name : Nom d'affichage
        - bio : Biographie
        - persona : Persona détaillée
        - age : Âge (entier)
        - gender : "male", "female" ou "other"
        - mbti : Type MBTI
        - country : Pays
        """
        data = []
        for idx, profile in enumerate(profiles):
            # Utiliser un format cohérent avec to_reddit_format()
            item = {
                "user_id": profile.user_id if profile.user_id is not None else idx,  # Clé : doit inclure user_id
                "username": profile.user_name,
                "name": profile.name,
                "bio": str(profile.bio)[:150] if profile.bio else f"{profile.name}",
                "persona": profile.persona or f"{profile.name} est un(e) participant(e) aux discussions sociales.",
                "karma": profile.karma if profile.karma else 1000,
                "created_at": profile.created_at,
                # Champs requis par OASIS - s'assurer que tous ont des valeurs par défaut
                "age": profile.age if profile.age else 30,
                "gender": self._normalize_gender(profile.gender),
                "mbti": profile.mbti if profile.mbti else "ISTJ",
                "country": profile.country if profile.country else "US",
            }

            # Champs optionnels
            if profile.profession:
                item["profession"] = profile.profession
            if profile.interested_topics:
                item["interested_topics"] = profile.interested_topics

            data.append(item)

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Sauvegardé {len(profiles)} profils Reddit dans {file_path} (format JSON, inclut le champ user_id)")
    
    # Garder l'ancien nom de méthode comme alias pour la rétrocompatibilité
    def save_profiles_to_json(
        self,
        profiles: List[OasisAgentProfile],
        file_path: str,
        platform: str = "reddit"
    ):
        """[Déprécié] Veuillez utiliser la méthode save_profiles()"""
        logger.warning("save_profiles_to_json est déprécié, veuillez utiliser la méthode save_profiles")
        self.save_profiles(profiles, file_path, platform)

