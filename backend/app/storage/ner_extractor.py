"""
Extracteur NER/RE — extraction d'entités et de relations via LLM local

Remplace le pipeline NER/RE intégré de Zep Cloud.
Utilise LLMClient.chat_json() avec un prompt structuré pour extraire
les entités et les relations des morceaux de texte, guidé par l'ontologie du graphe.
"""

import logging
from typing import Dict, Any, List, Optional

from ..utils.llm_client import LLMClient

logger = logging.getLogger('mirofish.ner_extractor')

# Modèle de prompt système pour l'extraction NER/RE
_SYSTEM_PROMPT = """You are a Named Entity Recognition and Relation Extraction system.
Given a text and an ontology (entity types + relation types), extract all entities and relations.

ONTOLOGY:
{ontology_description}

RULES:
1. Only extract entity types and relation types defined in the ontology.
2. Normalize entity names: strip whitespace, use canonical form (e.g., "Jack Ma" not "ma jack").
3. Each entity must have: name, type (from ontology), and optional attributes.
4. Each relation must have: source entity name, target entity name, type (from ontology), and a fact sentence describing the relationship.
5. If no entities or relations are found, return empty lists.
6. Be precise — only extract what is explicitly stated or strongly implied in the text.

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "...", "type": "...", "attributes": {{"key": "value"}}}}
  ],
  "relations": [
    {{"source": "...", "target": "...", "type": "...", "fact": "..."}}
  ]
}}"""

_USER_PROMPT = """Extract entities and relations from the following text:

{text}"""


class NERExtractor:
    """Extraire les entités et les relations du texte en utilisant un LLM local."""

    def __init__(self, llm_client: Optional[LLMClient] = None, max_retries: int = 2):
        self.llm = llm_client or LLMClient()
        self.max_retries = max_retries

    def extract(self, text: str, ontology: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extraire les entités et les relations du texte, guidé par l'ontologie.

        Args:
            text: Morceau de texte en entrée
            ontology: Dictionnaire avec 'entity_types' et 'relation_types' du graphe

        Returns:
            Dictionnaire avec les listes 'entities' et 'relations' :
            {
                "entities": [{"name": str, "type": str, "attributes": dict}],
                "relations": [{"source": str, "target": str, "type": str, "fact": str}]
            }
        """
        if not text or not text.strip():
            return {"entities": [], "relations": []}

        ontology_desc = self._format_ontology(ontology)
        system_msg = _SYSTEM_PROMPT.format(ontology_description=ontology_desc)
        user_msg = _USER_PROMPT.format(text=text.strip())

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                result = self.llm.chat_json(
                    messages=messages,
                    temperature=0.1,  # Température basse pour la précision de l'extraction
                    max_tokens=4096,
                )
                return self._validate_and_clean(result, ontology)

            except ValueError as e:
                last_error = e
                logger.warning(
                    f"L'extraction NER a échoué (tentative {attempt + 1}) : JSON invalide — {e}"
                )
            except Exception as e:
                last_error = e
                logger.error(f"Erreur d'extraction NER : {e}")
                if attempt >= self.max_retries:
                    break

        logger.error(
            f"L'extraction NER a échoué après {self.max_retries + 1} tentatives : {last_error}"
        )
        return {"entities": [], "relations": []}

    def _format_ontology(self, ontology: Dict[str, Any]) -> str:
        """Formater le dictionnaire d'ontologie en texte lisible pour le prompt du LLM."""
        parts = []

        entity_types = ontology.get("entity_types", [])
        if entity_types:
            parts.append("Entity Types:")
            for et in entity_types:
                if isinstance(et, dict):
                    name = et.get("name", str(et))
                    desc = et.get("description", "")
                    attrs = et.get("attributes", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if attrs:
                        attr_names = [a.get("name", str(a)) if isinstance(a, dict) else str(a) for a in attrs]
                        line += f" (attributes: {', '.join(attr_names)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {et}")

        relation_types = ontology.get("relation_types", ontology.get("edge_types", []))
        if relation_types:
            parts.append("\nRelation Types:")
            for rt in relation_types:
                if isinstance(rt, dict):
                    name = rt.get("name", str(rt))
                    desc = rt.get("description", "")
                    source_targets = rt.get("source_targets", [])
                    line = f"  - {name}"
                    if desc:
                        line += f": {desc}"
                    if source_targets:
                        st_strs = [f"{st.get('source', '?')} → {st.get('target', '?')}" for st in source_targets]
                        line += f" ({', '.join(st_strs)})"
                    parts.append(line)
                else:
                    parts.append(f"  - {rt}")

        if not parts:
            parts.append("No specific ontology defined. Extract all entities and relations you find.")

        return "\n".join(parts)

    def _validate_and_clean(
        self, result: Dict[str, Any], ontology: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Valider et normaliser la sortie du LLM."""
        entities = result.get("entities", [])
        relations = result.get("relations", [])

        # Obtenir les noms de types valides depuis l'ontologie
        valid_entity_types = set()
        for et in ontology.get("entity_types", []):
            if isinstance(et, dict):
                valid_entity_types.add(et.get("name", "").strip())
            else:
                valid_entity_types.add(str(et).strip())

        valid_relation_types = set()
        for rt in ontology.get("relation_types", ontology.get("edge_types", [])):
            if isinstance(rt, dict):
                valid_relation_types.add(rt.get("name", "").strip())
            else:
                valid_relation_types.add(str(rt).strip())

        # Nettoyer les entités
        cleaned_entities = []
        seen_names = set()
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            name = str(entity.get("name", "")).strip()
            etype = str(entity.get("type", "Entity")).strip()
            if not name:
                continue

            # Dédupliquer par nom normalisé
            name_lower = name.lower()
            if name_lower in seen_names:
                continue
            seen_names.add(name_lower)

            # Si l'ontologie a des types, avertir mais garder les entités avec des types inconnus
            if valid_entity_types and etype not in valid_entity_types:
                logger.debug(f"L'entité '{name}' a le type '{etype}' qui n'est pas dans l'ontologie, conservée quand même")

            cleaned_entities.append({
                "name": name,
                "type": etype,
                "attributes": entity.get("attributes", {}),
            })

        # Nettoyer les relations
        cleaned_relations = []
        entity_names_lower = {e["name"].lower() for e in cleaned_entities}
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            source = str(relation.get("source", "")).strip()
            target = str(relation.get("target", "")).strip()
            rtype = str(relation.get("type", "RELATED_TO")).strip()
            fact = str(relation.get("fact", "")).strip()

            if not source or not target:
                continue

            # S'assurer que les entités source et cible existent
            # (elles peuvent ne pas exister si le LLM a halluciné une relation sans l'entité)
            if source.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": source,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(source.lower())

            if target.lower() not in entity_names_lower:
                cleaned_entities.append({
                    "name": target,
                    "type": "Entity",
                    "attributes": {},
                })
                entity_names_lower.add(target.lower())

            cleaned_relations.append({
                "source": source,
                "target": target,
                "type": rtype,
                "fact": fact or f"{source} {rtype} {target}",
            })

        return {
            "entities": cleaned_entities,
            "relations": cleaned_relations,
        }
