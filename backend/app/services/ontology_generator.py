"""
Service de génération d'ontologie
Interface 1 : Analyser le contenu textuel et générer des définitions de types d'entités et de relations adaptées à la simulation sociale
"""

import json
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient


# Prompt système pour la génération d'ontologie
ONTOLOGY_SYSTEM_PROMPT = """Vous êtes un expert professionnel en conception d'ontologie de graphe de connaissances. Votre tâche est d'analyser le contenu textuel et les exigences de simulation donnés, et de concevoir des types d'entités et de relations adaptés à la **simulation d'opinions sur les médias sociaux**.

**Important : Vous devez produire des données au format JSON valide, ne produisez rien d'autre.**

## Contexte de la tâche principale

Nous construisons un **système de simulation d'opinions sur les médias sociaux**. Dans ce système :
- Chaque entité est un « compte » ou un « sujet » pouvant s'exprimer, interagir et diffuser des informations sur les médias sociaux
- Les entités s'influencent mutuellement, retweetent, commentent et répondent
- Nous devons simuler les réactions des différentes parties lors d'événements d'opinion et les chemins de diffusion de l'information

Par conséquent, **les entités doivent être des entités du monde réel pouvant s'exprimer et interagir sur les médias sociaux** :

**Peuvent être** :
- Des individus spécifiques (personnalités publiques, parties prenantes, leaders d'opinion, experts, personnes ordinaires)
- Des entreprises et sociétés (y compris leurs comptes officiels)
- Des organisations (universités, associations, ONG, syndicats, etc.)
- Des départements gouvernementaux et agences de réglementation
- Des institutions médiatiques (journaux, chaînes de télévision, médias indépendants, sites web)
- Les plateformes de médias sociaux elles-mêmes
- Des représentants de groupes spécifiques (comme les associations d'anciens élèves, les groupes de fans, les groupes de défense des droits, etc.)

**Ne peuvent pas être** :
- Des concepts abstraits (comme « l'opinion publique », « l'émotion », « la tendance »)
- Des sujets/thèmes (comme « l'intégrité académique », « la réforme de l'éducation »)
- Des vues/attitudes (comme « les partisans », « les opposants »)

## Format de sortie

Veuillez produire au format JSON avec la structure suivante :

```json
{
    "entity_types": [
        {
            "name": "Nom du type d'entité (anglais, PascalCase)",
            "description": "Brève description (anglais, pas plus de 100 caractères)",
            "attributes": [
                {
                    "name": "Nom de l'attribut (anglais, snake_case)",
                    "type": "text",
                    "description": "Description de l'attribut"
                }
            ],
            "examples": ["Exemple d'entité 1", "Exemple d'entité 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Nom du type de relation (anglais, UPPER_SNAKE_CASE)",
            "description": "Brève description (anglais, pas plus de 100 caractères)",
            "source_targets": [
                {"source": "Type d'entité source", "target": "Type d'entité cible"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Brève analyse et explication du contenu textuel"
}
```

## Directives de conception (Extrêmement important !)

### 1. Conception des types d'entités — À suivre strictement

**Exigence de quantité : Doit avoir exactement 10 types d'entités**

**Exigence de structure hiérarchique (doit inclure à la fois des types spécifiques et des types de secours)** :

Vos 10 types d'entités doivent inclure la hiérarchie suivante :

A. **Types de secours (à inclure obligatoirement, placer en derniers dans la liste)** :
   - `Person` : Type de secours pour toute personne physique. Quand une personne ne correspond pas à d'autres types de personnes plus spécifiques, utilisez celui-ci.
   - `Organization` : Type de secours pour toute organisation. Quand une organisation ne correspond pas à d'autres types d'organisations plus spécifiques, utilisez celui-ci.

B. **Types spécifiques (8, conçus en fonction du contenu textuel)** :
   - Concevoir des types plus spécifiques pour les personnages principaux apparaissant dans le texte
   - Exemple : Si le texte concerne des événements académiques, peut avoir `Student`, `Professor`, `University`
   - Exemple : Si le texte concerne des événements commerciaux, peut avoir `Company`, `CEO`, `Employee`

**Pourquoi les types de secours sont nécessaires** :
- Diverses personnes apparaîtront dans le texte, comme des « enseignants du primaire/secondaire », des « personnes quelconques », des « internautes »
- Si aucun type spécifique ne correspond, ils doivent être classés comme `Person`
- De même, les petites organisations et groupes temporaires doivent être classés comme `Organization`

**Principes de conception pour les types spécifiques** :
- Identifier les types de rôles à haute fréquence ou clés dans le texte
- Chaque type spécifique doit avoir des frontières claires, éviter les chevauchements
- La description doit expliquer clairement la différence entre ce type et le type de secours

### 2. Conception des types de relations

- Quantité : 6-10
- Les relations doivent refléter des connexions réelles dans les interactions sur les médias sociaux
- S'assurer que les source_targets des relations couvrent les types d'entités que vous avez définis

### 3. Conception des attributs

- 1-3 attributs clés par type d'entité
- **Note** : Les noms d'attributs ne peuvent pas utiliser `name`, `uuid`, `group_id`, `created_at`, `summary` (ce sont des mots réservés du système)
- Recommandé : `full_name`, `title`, `role`, `position`, `location`, `description`, etc.

## Référence des types d'entités

**Types individuels (spécifiques)** :
- Student : Étudiant
- Professor : Professeur/Chercheur
- Journalist : Journaliste
- Celebrity : Célébrité/Influenceur
- Executive : Cadre dirigeant
- Official : Fonctionnaire
- Lawyer : Avocat
- Doctor : Médecin

**Types individuels (secours)** :
- Person : Toute personne physique (à utiliser quand ne correspond pas à d'autres types spécifiques)

**Types d'organisation (spécifiques)** :
- University : Université
- Company : Entreprise/Société
- GovernmentAgency : Agence gouvernementale
- MediaOutlet : Institution médiatique
- Hospital : Hôpital
- School : École primaire/secondaire
- NGO : Organisation non gouvernementale

**Types d'organisation (secours)** :
- Organization : Toute organisation (à utiliser quand ne correspond pas à d'autres types spécifiques)

## Référence des types de relations

- WORKS_FOR : Travaille pour
- STUDIES_AT : Étudie à
- AFFILIATED_WITH : Affilié à
- REPRESENTS : Représente
- REGULATES : Réglemente
- REPORTS_ON : Rapporte sur
- COMMENTS_ON : Commente
- RESPONDS_TO : Répond à
- SUPPORTS : Soutient
- OPPOSES : S'oppose à
- COLLABORATES_WITH : Collabore avec
- COMPETES_WITH : Est en concurrence avec
"""


class OntologyGenerator:
    """
    Ontology generator
    Analyser le contenu textuel et générer des définitions de types d'entités et de relations
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()

    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Générer la définition d'ontologie

        Args:
            document_texts: Liste des textes de documents
            simulation_requirement: Description des exigences de simulation
            additional_context: Contexte additionnel

        Returns:
            Définition d'ontologie (entity_types, edge_types, etc.)
        """
        # Build user message
        user_message = self._build_user_message(
            document_texts,
            simulation_requirement,
            additional_context
        )

        messages = [
            {"role": "system", "content": ONTOLOGY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        # Call LLM
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )

        # Validate and post-process
        result = self._validate_and_process(result)

        return result

    # Maximum text length for LLM (50,000 characters)
    MAX_TEXT_LENGTH_FOR_LLM = 50000

    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Construire le message utilisateur"""

        # Combiner les textes
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)

        # Si le texte dépasse 50 000 caractères, tronquer (affecte uniquement l'entrée du LLM, pas la construction du graphe)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Le texte original a {original_length} caractères, les {self.MAX_TEXT_LENGTH_FOR_LLM} premiers caractères extraits pour l'analyse d'ontologie)..."

        message = f"""## Exigences de simulation

{simulation_requirement}

## Contenu du document

{combined_text}
"""

        if additional_context:
            message += f"""
## Explication supplémentaire

{additional_context}
"""

        message += """
Sur la base du contenu ci-dessus, concevez des types d'entités et de relations adaptés à la simulation d'opinions sociales.

**Règles à suivre** :
1. Doit produire exactement 10 types d'entités
2. Les 2 derniers doivent être des types de secours : Person (secours individuel) et Organization (secours organisation)
3. Les 8 premiers sont des types spécifiques conçus en fonction du contenu textuel
4. Tous les types d'entités doivent être des sujets du monde réel pouvant s'exprimer, pas des concepts abstraits
5. Les noms d'attributs ne peuvent pas utiliser des mots réservés comme name, uuid, group_id, utilisez full_name, org_name, etc. à la place
"""

        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Valider et post-traiter le résultat"""

        # S'assurer que les champs nécessaires existent
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""

        # Valider les types d'entités
        for entity in result["entity_types"]:
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # S'assurer que la description ne dépasse pas 100 caractères
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."

        # Valider les types de relations
        for edge in result["edge_types"]:
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."

        # Limite de l'API : maximum 10 types d'entités personnalisés, maximum 10 types de relations personnalisés
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # Définitions des types de secours
        person_fallback = {
            "name": "Person",
            "description": "Toute personne physique ne correspondant pas à d'autres types de personnes spécifiques.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Nom complet de la personne"},
                {"name": "role", "type": "text", "description": "Rôle ou profession"}
            ],
            "examples": ["citoyen ordinaire", "internaute anonyme"]
        }

        organization_fallback = {
            "name": "Organization",
            "description": "Toute organisation ne correspondant pas à d'autres types d'organisations spécifiques.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Nom de l'organisation"},
                {"name": "org_type", "type": "text", "description": "Type d'organisation"}
            ],
            "examples": ["petite entreprise", "groupe communautaire"]
        }

        # Vérifier si les types de secours existent déjà
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names

        # Types de secours à ajouter
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)

        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)

            # Si l'ajout dépasserait 10, il faut retirer certains types existants
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Calculer combien retirer
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Retirer de la fin (garder les types spécifiques plus importants en premier)
                result["entity_types"] = result["entity_types"][:-to_remove]

            # Ajouter les types de secours
            result["entity_types"].extend(fallbacks_to_add)

        # Vérification finale pour s'assurer que les limites ne sont pas dépassées (programmation défensive)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]

        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]

        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        [DÉPRÉCIÉ] Convertir la définition d'ontologie en code Pydantic au format Zep.
        Non utilisé dans MiroFish-Offline (ontologie stockée en JSON dans Neo4j).
        Conservé pour référence uniquement.
        """
        code_lines = [
            '"""',
            "Définitions de types d'entités personnalisés",
            "Auto-généré par MiroFish pour la simulation d'opinions sociales",
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            "# ============== Définitions des types d'entités ==============",
            '',
        ]

        # Générer les types d'entités
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")

            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        code_lines.append('# ============== Définitions des types de relations ==============')
        code_lines.append('')

        # Générer les types de relations
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # Convert to PascalCase class name
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")

            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')

            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')

            code_lines.append('')
            code_lines.append('')

        # Générer les dictionnaires de types
        code_lines.append('# ============== Configuration des types ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')

        # Générer le mappage source_targets pour les arêtes
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')

        return '\n'.join(code_lines)

