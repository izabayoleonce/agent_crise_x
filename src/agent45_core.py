# -*- coding: utf-8 -*-
"""
CrisisAI War Room — Agents 4 et 5
=================================
Agent 4 : Stratège de riposte
Agent 5 : Rédacteur + garde-fou

Ces agents sont branchés sur le résultat de l'Agent 1. Ils restent prudents :
- pas de publication automatique ;
- pas d'accusation de bots sans preuve ;
- pas de chiffres inventés ;
- recommandation validable par une cellule humaine.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .agent1_core import crisis_level, format_int, result_context_json


# -----------------------------------------------------------------------------
# Helpers de contexte
# -----------------------------------------------------------------------------


def _safe_records(df: Any, limit: int = 8) -> List[Dict[str, Any]]:
    if isinstance(df, pd.DataFrame) and len(df):
        small = df.head(limit).copy()
        for col in small.columns:
            if pd.api.types.is_datetime64_any_dtype(small[col]):
                small[col] = small[col].astype(str)
        return small.to_dict(orient="records")
    return []


def _first_value(records: List[Dict[str, Any]], key: str, default: str = "non identifié") -> Any:
    if not records:
        return default
    return records[0].get(key, default)


def build_crisis_context(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convertit le résultat Agent 1 en contexte compact pour Agents 4/5."""
    kpis = result.get("kpis", {})
    score = float(result.get("crisis_velocity_score", 0) or 0)
    narratives = _safe_records(result.get("narratives"), limit=8)
    top_authors = _safe_records(result.get("top_authors"), limit=8)
    top_posts = _safe_records(result.get("top_posts"), limit=5)
    dominant_narrative = _first_value(narratives, "narrative")
    top_author = _first_value(top_authors, "author")

    return {
        "periode": result.get("periode", {}),
        "kpis": kpis,
        "score": score,
        "niveau": crisis_level(score),
        "dominant_narrative": dominant_narrative,
        "top_author": top_author,
        "narratives": narratives,
        "top_authors": top_authors,
        "top_posts": top_posts,
        "brief_agent1": result.get("brief_deterministe", ""),
    }


def priority_from_score(score: float) -> str:
    if score >= 80:
        return "P1 — Crise forte : cellule de crise active, validation direction, suivi horaire"
    if score >= 60:
        return "P2 — Crise élevée : réponse préparée, monitoring renforcé, validation communication"
    if score >= 40:
        return "P3 — Crise modérée : clarification possible, surveillance des relais"
    return "P4 — Signal faible : monitoring, pas de réponse publique immédiate sauf sollicitation"


def response_posture_from_context(context: Dict[str, Any]) -> str:
    dominant = str(context.get("dominant_narrative", "")).lower()
    score = context.get("score", 0)
    if "harc" in dominant or "menace" in dominant:
        return "Empathique + sécurité : condamner les menaces, rappeler les faits, protéger les personnes"
    if "argent" in dominant or "subvention" in dominant or "copinage" in dominant:
        return "Transparence + pédagogie : expliquer les règles, critères, garde-fous et gouvernance"
    if "censure" in dominant or "idéologie" in dominant or "ideologie" in dominant:
        return "Neutralité institutionnelle : rappeler les principes sans entrer dans l'affrontement politique"
    if score >= 60:
        return "Factuel + apaisement : reconnaître l'inquiétude, clarifier, éviter de sur-réagir"
    return "Monitoring + réponse réactive : répondre seulement aux sollicitations qualifiées"


def _dominant_narrative_sentence(context: Dict[str, Any]) -> str:
    n = context.get("dominant_narrative", "non identifié")
    return f"Le narratif dominant détecté est : {n}."


def _safe_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "n/a"


# -----------------------------------------------------------------------------
# Agent 4 : Stratège de riposte
# -----------------------------------------------------------------------------


@dataclass
class AgentStrategieRiposte:
    """Transforme le diagnostic Agent 1 en stratégie de réponse."""

    def build_strategy(
        self,
        result: Dict[str, Any],
        objective: str = "Rétablir la confiance et clarifier les faits",
        posture: str = "Automatique selon la crise",
        cible: Optional[Iterable[str]] = None,
        risk_tolerance: str = "Prudente",
        response_window: str = "Prochaines 24 heures",
        agent2_report: Optional[Dict[str, Any]] = None,
        agent3_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context = build_crisis_context(result)
        # Connexion Top 1 : Agent 4 peut exploiter les sorties Agents 2 et 3
        # sans casser la compatibilité avec l'ancien pipeline Agent 1 -> Agent 4.
        if agent2_report:
            context["agent2_summary"] = {
                "top_narratives": _safe_records(agent2_report.get("risk_matrix"), limit=5),
                "mutations": _safe_records(agent2_report.get("mutation_signals"), limit=5),
                "markdown": agent2_report.get("markdown", ""),
            }
        if agent3_report:
            coord = agent3_report.get("coordination_score", {})
            context["agent3_summary"] = {
                "coordination_score": coord,
                "propagation_metrics": agent3_report.get("propagation_metrics", {}),
                "markdown": agent3_report.get("markdown", ""),
            }
        k = context.get("kpis", {})
        score = context.get("score", 0)
        cible_list = list(cible or ["Grand public", "Médias", "Créateurs concernés"])
        chosen_posture = response_posture_from_context(context) if posture == "Automatique selon la crise" else posture

        sensitive = self._sensitive_points(context)
        actions = self._action_plan(context, chosen_posture, response_window)
        messages = self._message_axes(context, chosen_posture)
        avoid = self._avoid_list(context)
        escalation = self._escalation_triggers(context)

        strategy = {
            "agent": "Agent 4 — Stratège de riposte",
            "objective": objective,
            "cible": cible_list,
            "risk_tolerance": risk_tolerance,
            "response_window": response_window,
            "priority": priority_from_score(score),
            "posture": chosen_posture,
            "context": context,
            "sensitive_points": sensitive,
            "message_axes": messages,
            "action_plan": actions,
            "avoid": avoid,
            "risks_if_silent": self._risks_if_silent(context),
            "risks_if_overreacting": self._risks_if_overreacting(context),
            "escalation_triggers": escalation,
        }
        strategy["markdown"] = self.to_markdown(strategy)
        return strategy

    def _sensitive_points(self, context: Dict[str, Any]) -> List[str]:
        pts = [
            "Ne pas traiter le dossier comme une simple fake news : les faits de départ sont datés et vérifiables.",
            "Le cœur du risque est la dynamique virale et la perte de confiance institutionnelle.",
        ]
        dominant = str(context.get("dominant_narrative", "")).lower()
        if any(x in dominant for x in ["argent", "subvention", "copinage", "favoritisme"]):
            pts.append("Le narratif attaque la légitimité des aides publiques : réponse nécessaire sur transparence, critères et gouvernance.")
        if "censure" in dominant or "idéologie" in dominant or "ideologie" in dominant:
            pts.append("Le sujet peut être recadré politiquement : éviter tout vocabulaire partisan.")
        if "harc" in dominant or "menace" in dominant:
            pts.append("Risque de sécurité/réputation : condamner les menaces sans nier les critiques légitimes.")
        return pts

    def _message_axes(self, context: Dict[str, Any], posture: str) -> List[Dict[str, str]]:
        return [
            {
                "axe": "Clarification factuelle",
                "but": "Recentrer la discussion sur les faits vérifiables et le périmètre exact de la décision.",
                "formulation": "Rappeler le calendrier, la décision prise et les règles applicables sans commenter les personnes.",
            },
            {
                "axe": "Transparence des règles",
                "but": "Répondre au narratif argent public/copinerie sans agressivité.",
                "formulation": "Expliquer les critères d'attribution, les procédures de sélection et les garde-fous existants.",
            },
            {
                "axe": "Apaisement",
                "but": "Réduire la polarisation et éviter de nourrir la polémique.",
                "formulation": "Reconnaître les interrogations du public et s'engager à fournir des informations claires.",
            },
            {
                "axe": "Human-in-the-loop",
                "but": "Montrer que l'IA assiste mais ne décide pas.",
                "formulation": "Tout message proposé doit être validé par la cellule communication avant publication.",
            },
        ]

    def _action_plan(self, context: Dict[str, Any], posture: str, response_window: str) -> List[Dict[str, str]]:
        return [
            {
                "temps": "0–1 h",
                "action": "Stabiliser le diagnostic",
                "détail": "Valider le pic, les narratifs dominants, les comptes moteurs et les publications les plus engageantes.",
            },
            {
                "temps": "1–3 h",
                "action": "Préparer les éléments de langage",
                "détail": "Rédiger une clarification courte + une FAQ interne sur critères, gouvernance et calendrier.",
            },
            {
                "temps": "3–6 h",
                "action": "Choisir le canal de réponse",
                "détail": "Publier seulement si le narratif continue de monter ou si des médias/influenceurs structurants relaient.",
            },
            {
                "temps": "6–24 h",
                "action": "Suivre la mutation narrative",
                "détail": "Surveiller si la crise passe d'Ultia vers argent public, censure, sécurité ou défiance institutionnelle.",
            },
        ]

    def _avoid_list(self, context: Dict[str, Any]) -> List[str]:
        return [
            "Ne pas attaquer les internautes, journalistes, créateurs ou camps politiques.",
            "Ne pas minimiser la colère : préférer reconnaître les interrogations.",
            "Ne pas employer de termes accusatoires non prouvés comme 'bots' ou 'manipulation organisée'.",
            "Ne pas publier un message long et défensif sur X en premier réflexe.",
            "Ne pas promettre une enquête, une sanction ou une réforme si ce n'est pas validé en interne.",
        ]

    def _risks_if_silent(self, context: Dict[str, Any]) -> List[str]:
        return [
            "Le vide de communication peut laisser le narratif dominant s'installer.",
            "Les comptes moteurs peuvent imposer leur cadrage sans contradiction factuelle.",
            "Le débat peut muter vers une critique plus large de l'institution et des aides publiques.",
        ]

    def _risks_if_overreacting(self, context: Dict[str, Any]) -> List[str]:
        return [
            "Une réponse trop défensive peut relancer le pic et générer de nouveaux quote tweets.",
            "Une réponse politique peut confirmer le narratif de partialité/censure.",
            "Une réponse trop juridique peut paraître froide ou opaque.",
        ]

    def _escalation_triggers(self, context: Dict[str, Any]) -> List[str]:
        return [
            "Nouveau pic horaire supérieur au pic précédent ou reprise par un média national.",
            "Hausse du narratif 'argent public/copinerie' sur plusieurs heures.",
            "Apparition de menaces, doxxing, appels ciblés ou risque de sécurité.",
            "Interpellation directe d'un décideur, ministère, partenaire ou média majeur.",
        ]

    def to_markdown(self, strategy: Dict[str, Any]) -> str:
        ctx = strategy["context"]
        k = ctx.get("kpis", {})
        lines: List[str] = []
        lines.append("## Agent 4 — Stratégie de riposte")
        lines.append("")
        lines.append(f"**Période source :** {ctx.get('periode', {}).get('start', 'n/a')} → {ctx.get('periode', {}).get('end', 'n/a')}")
        lines.append(f"**Niveau :** {ctx.get('niveau')} — score {ctx.get('score')}/100")
        lines.append(f"**Priorité :** {strategy['priority']}")
        lines.append(f"**Objectif :** {strategy['objective']}")
        lines.append(f"**Posture recommandée :** {strategy['posture']}")
        lines.append(f"**Cibles :** {', '.join(strategy['cible'])}")
        lines.append("")
        lines.append("### Diagnostic stratégique")
        lines.append(f"- Messages analysés : {format_int(k.get('messages', 0))} ; retweets/reposts : {_safe_pct(k.get('retweets_pct', 0))}.")
        lines.append(f"- {_dominant_narrative_sentence(ctx)}")
        lines.append("- Le risque prioritaire est la perte de contrôle du cadrage public, plus que la simple correction d'un fait.")
        if ctx.get("agent2_summary"):
            top_ns = ctx["agent2_summary"].get("top_narratives", [])
            if top_ns:
                lines.append(f"- Agent 2 confirme le narratif prioritaire : **{top_ns[0].get('narrative', 'n/a')}** avec un risque {top_ns[0].get('risk_label', 'n/a')}.")
        if ctx.get("agent3_summary"):
            coord = ctx["agent3_summary"].get("coordination_score", {})
            lines.append(f"- Agent 3 mesure la propagation/coordination prudente : **{coord.get('score', 0)}/100** — {coord.get('label', 'n/a')}.")
        lines.append("")
        lines.append("### Axes de réponse")
        for item in strategy["message_axes"]:
            lines.append(f"- **{item['axe']}** — {item['but']} {item['formulation']}")
        lines.append("")
        lines.append("### Plan d'action")
        for item in strategy["action_plan"]:
            lines.append(f"- **{item['temps']} — {item['action']} :** {item['détail']}")
        lines.append("")
        lines.append("### À éviter absolument")
        for item in strategy["avoid"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("### Déclencheurs d'escalade")
        for item in strategy["escalation_triggers"]:
            lines.append(f"- {item}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Agent 5 : Rédacteur + garde-fou
# -----------------------------------------------------------------------------


@dataclass
class AgentRedacteurGardeFou:
    """Produit des messages validables et vérifie les risques de communication."""

    def generate_pack(
        self,
        result: Dict[str, Any],
        strategy: Dict[str, Any],
        tone: str = "Institutionnel, clair et apaisant",
        channels: Optional[Iterable[str]] = None,
        spokesperson: str = "Le CNC",
    ) -> Dict[str, Any]:
        context = build_crisis_context(result)
        channels_list = list(channels or ["Post X court", "Communiqué", "FAQ", "Réponse journaliste", "Message interne"])
        messages: Dict[str, str] = {}
        if "Post X court" in channels_list:
            messages["Post X court"] = self._post_x_short(context, strategy, spokesperson)
        if "Thread X" in channels_list:
            messages["Thread X"] = self._thread_x(context, strategy, spokesperson)
        if "Communiqué" in channels_list:
            messages["Communiqué"] = self._communique(context, strategy, spokesperson)
        if "FAQ" in channels_list:
            messages["FAQ"] = self._faq(context, strategy, spokesperson)
        if "Réponse journaliste" in channels_list:
            messages["Réponse journaliste"] = self._journalist_response(context, strategy, spokesperson)
        if "Message interne" in channels_list:
            messages["Message interne"] = self._internal_message(context, strategy, spokesperson)
        if "Message d'attente" in channels_list:
            messages["Message d'attente"] = self._holding_statement(context, strategy, spokesperson)

        guardrail = self.guardrail_check(messages, context)
        pack = {
            "agent": "Agent 5 — Rédacteur + garde-fou",
            "tone": tone,
            "channels": channels_list,
            "spokesperson": spokesperson,
            "context": context,
            "strategy_summary": strategy.get("markdown", ""),
            "messages": messages,
            "guardrail": guardrail,
        }
        pack["markdown"] = self.to_markdown(pack)
        return pack

    def _post_x_short(self, context: Dict[str, Any], strategy: Dict[str, Any], spokesperson: str) -> str:
        dominant = context.get("dominant_narrative", "la situation")
        return (
            f"{spokesperson} prend acte des interrogations suscitées par cette situation. "
            "Notre priorité est de rappeler les faits, les règles applicables et les garanties d'impartialité. "
            "Des éléments de clarification seront partagés de manière transparente, dans un cadre factuel et apaisé."
        )

    def _thread_x(self, context: Dict[str, Any], strategy: Dict[str, Any], spokesperson: str) -> str:
        return (
            "1/ Plusieurs questions circulent au sujet du dispositif d'aide concerné. Nous souhaitons clarifier les faits sans alimenter la polémique.\n\n"
            "2/ Les décisions d'attribution reposent sur des critères et procédures encadrés. Les règles de sélection et les garanties d'impartialité doivent rester le point central du débat.\n\n"
            "3/ Nous condamnons toute menace ou attaque personnelle. Les critiques peuvent être entendues, mais elles doivent rester dans un cadre respectueux.\n\n"
            "4/ Une synthèse factuelle sera mise à disposition afin de répondre aux principales interrogations."
        )

    def _communique(self, context: Dict[str, Any], strategy: Dict[str, Any], spokesperson: str) -> str:
        score = context.get("score", 0)
        level = context.get("niveau", "n/a")
        return (
            f"**Communiqué — {spokesperson}**\n\n"
            "Depuis plusieurs heures, de nombreuses réactions circulent sur les réseaux sociaux au sujet d'un dispositif d'aide. "
            "Nous comprenons que cette situation suscite des interrogations, notamment sur les règles d'attribution, l'impartialité des commissions et l'usage des fonds publics.\n\n"
            "Notre réponse se veut factuelle : les procédures doivent être lisibles, vérifiables et compréhensibles par le public. "
            "Les éléments nécessaires seront rappelés de manière transparente, sans personnalisation du débat et sans prise de position partisane.\n\n"
            "Nous condamnons par ailleurs toute menace, attaque personnelle ou forme de harcèlement. "
            "La discussion publique doit pouvoir se tenir dans un cadre respectueux.\n\n"
            "Notre priorité est de préserver la confiance, de clarifier les règles et de permettre une compréhension précise du fonctionnement du dispositif."
        )

    def _faq(self, context: Dict[str, Any], strategy: Dict[str, Any], spokesperson: str) -> str:
        return (
            "**FAQ — éléments de réponse**\n\n"
            "**1. Pourquoi communiquer maintenant ?**\n"
            "Parce que le volume de réactions et la vitesse de propagation justifient une clarification factuelle.\n\n"
            "**2. Quel est le point central ?**\n"
            "Le point central est la compréhension des règles, des critères d'attribution et des garanties d'impartialité.\n\n"
            "**3. Est-ce une fake news ?**\n"
            "Le sujet ne doit pas être réduit à une fake news. Il s'agit d'une dynamique virale autour de faits, d'interprétations et de narratifs concurrents.\n\n"
            "**4. Que faut-il éviter ?**\n"
            "Éviter les attaques personnelles, les formulations politiques et toute accusation non vérifiée.\n\n"
            "**5. Qui valide la réponse ?**\n"
            "Les messages proposés par l'IA doivent être relus et validés par une cellule humaine avant toute publication."
        )

    def _journalist_response(self, context: Dict[str, Any], strategy: Dict[str, Any], spokesperson: str) -> str:
        return (
            "Bonjour,\n\n"
            "Nous confirmons suivre attentivement les réactions en cours. Notre position est de traiter ce sujet de manière factuelle, "
            "en rappelant les règles applicables, les critères de sélection et les garanties d'impartialité. "
            "Nous ne souhaitons pas personnaliser le débat ni l'inscrire dans un affrontement politique.\n\n"
            "Des éléments de clarification pourront être transmis dès validation par la cellule communication.\n\n"
            "Bien cordialement,"
        )

    def _internal_message(self, context: Dict[str, Any], strategy: Dict[str, Any], spokesperson: str) -> str:
        return (
            "**Message interne — cellule de crise**\n\n"
            f"Niveau détecté : {context.get('niveau')} — score {context.get('score')}/100.\n"
            f"Narratif dominant : {context.get('dominant_narrative')}.\n\n"
            "Priorité : stabiliser les faits, éviter les réactions individuelles et centraliser la validation des messages. "
            "Ne pas répondre directement aux interpellations sans validation. Surveiller les reprises médias, les menaces et les mutations de narratif."
        )

    def _holding_statement(self, context: Dict[str, Any], strategy: Dict[str, Any], spokesperson: str) -> str:
        return (
            "Nous avons pris connaissance des nombreuses réactions en cours. "
            "Une clarification factuelle est en préparation afin de répondre aux principales interrogations. "
            "Notre priorité est de fournir des informations précises, vérifiables et respectueuses du débat public."
        )

    def guardrail_check(self, messages: Dict[str, str], context: Dict[str, Any]) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        forbidden_patterns = {
            "attaque personnelle": r"\b(idiot|imbecile|menteur|mensonge|honte a vous|facho|gauchiste)\b",
            "accusation non prouvée": r"\b(bot|bots|manipulation organisée|complot|corruption prouvée|détournement prouvé)\b",
            "promesse non validée": r"\b(nous garantissons|nous promettons|enquête officielle immédiate|sanction immédiate)\b",
        }
        expected_patterns = {
            "factuel/transparence": r"\b(fait|factuel|transparen|règle|regle|critère|critere|procédure|procedure)\b",
            "apaisement": r"\b(apais|interrogation|respect|clarification|comprend|écoute|ecoute)\b",
            "validation humaine": r"\b(valid|cellule|communication|humain|relecture)\b",
        }
        for channel, text in messages.items():
            text_norm = text.lower()
            flags = []
            for label, pattern in forbidden_patterns.items():
                if re.search(pattern, text_norm, flags=re.IGNORECASE):
                    flags.append(f"risque : {label}")
            missing = []
            for label, pattern in expected_patterns.items():
                if not re.search(pattern, text_norm, flags=re.IGNORECASE):
                    missing.append(label)
            status = "À valider humainement"
            if flags:
                status = "À corriger avant validation"
            elif len(missing) <= 1:
                status = "Propre, mais validation humaine requise"
            rows.append({
                "canal": channel,
                "statut": status,
                "risques_detectés": "; ".join(flags) if flags else "Aucun risque majeur détecté",
                "points_à_renforcer": "; ".join(missing) if missing else "RAS",
                "longueur_caractères": len(text),
            })
        return pd.DataFrame(rows)

    def to_markdown(self, pack: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("## Agent 5 — Messages rédigés + garde-fou")
        lines.append("")
        lines.append(f"**Ton :** {pack.get('tone')}")
        lines.append(f"**Porte-parole :** {pack.get('spokesperson')}")
        lines.append("")
        for channel, msg in pack.get("messages", {}).items():
            lines.append(f"### {channel}")
            lines.append(msg)
            lines.append("")
        lines.append("### Garde-fou")
        guardrail = pack.get("guardrail")
        if isinstance(guardrail, pd.DataFrame):
            for _, row in guardrail.iterrows():
                lines.append(f"- **{row['canal']}** : {row['statut']} — {row['risques_detectés']}.")
        lines.append("")
        lines.append("**Règle finale :** aucun message n'est publié automatiquement. La cellule communication garde la décision finale.")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Exports Agents 4/5
# -----------------------------------------------------------------------------


def agent45_export_files(strategy: Optional[Dict[str, Any]] = None, draft_pack: Optional[Dict[str, Any]] = None) -> Dict[str, bytes]:
    files: Dict[str, bytes] = {}
    if strategy:
        files["agent4_strategie_riposte.md"] = strategy.get("markdown", "").encode("utf-8")
        strategy_json = {k: v for k, v in strategy.items() if k != "context"}
        files["agent4_strategie_riposte.json"] = json.dumps(strategy_json, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    if draft_pack:
        files["agent5_messages_gardefou.md"] = draft_pack.get("markdown", "").encode("utf-8")
        guardrail = draft_pack.get("guardrail")
        if isinstance(guardrail, pd.DataFrame):
            files["agent5_gardefou.csv"] = guardrail.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        messages = draft_pack.get("messages", {})
        files["agent5_messages.json"] = json.dumps(messages, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return files


def make_zip(files: Dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    buffer.seek(0)
    return buffer.getvalue()
