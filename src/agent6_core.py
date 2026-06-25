# -*- coding: utf-8 -*-
"""
CrisisAI War Room — Agent 6
===========================
Agent 6 : Chatbot de crise RAG-light

Il répond aux questions de l'utilisateur en s'appuyant en priorité sur les sorties
vérifiées des Agents 1 à 5 : diagnostic, narratifs, propagation, stratégie et
messages. Le LLM est optionnel : si OpenRouter est disponible, il reformule la
réponse à partir du contexte JSON ; sinon une réponse déterministe est produite.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from .agent1_core import format_int
from .agent45_core import build_crisis_context


def _safe_records(df: Any, limit: int = 8) -> List[Dict[str, Any]]:
    """Convertit proprement une DataFrame en records JSON compacts."""
    if isinstance(df, pd.DataFrame) and len(df):
        small = df.head(limit).copy()
        for col in small.columns:
            if pd.api.types.is_datetime64_any_dtype(small[col]):
                small[col] = small[col].astype(str)
        return small.to_dict(orient="records")
    return []


def _shorten(text: Any, limit: int = 4200) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n\n...[contenu tronqué pour rester compact]"


def _df_to_markdown(df: Any, limit: int = 8) -> str:
    if isinstance(df, pd.DataFrame) and len(df):
        return df.head(limit).to_markdown(index=False)
    return "Aucune donnée disponible."


def build_agent6_context(
    result: Dict[str, Any],
    agent2_report: Optional[Dict[str, Any]] = None,
    agent3_report: Optional[Dict[str, Any]] = None,
    strategy: Optional[Dict[str, Any]] = None,
    draft_pack: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Construit le contexte prioritaire de l'Agent 6 à partir des Agents 1 à 5."""
    base = build_crisis_context(result)

    context = {
        "agent1_diagnostic": base,
        "agent2_narratifs": {},
        "agent3_propagation": {},
        "agent4_strategie": {},
        "agent5_redaction": {},
        "source_priority": [
            "Agent 1 — diagnostic chiffré",
            "Agent 2 — narratifs / acteurs",
            "Agent 3 — propagation / coordination prudente",
            "Agent 4 — stratégie de riposte",
            "Agent 5 — messages / garde-fou",
        ],
    }

    if agent2_report:
        context["agent2_narratifs"] = {
            "markdown": _shorten(agent2_report.get("markdown", ""), 3600),
            "risk_matrix": _safe_records(agent2_report.get("risk_matrix"), 8),
            "actor_narratives": _safe_records(agent2_report.get("actor_narratives"), 8),
            "mutation_signals": _safe_records(agent2_report.get("mutation_signals"), 8),
            "recommendations": agent2_report.get("recommendations", []),
        }

    if agent3_report:
        context["agent3_propagation"] = {
            "markdown": _shorten(agent3_report.get("markdown", ""), 3600),
            "coordination_score": agent3_report.get("coordination_score", {}),
            "propagation_metrics": agent3_report.get("propagation_metrics", {}),
            "burst_patterns": _safe_records(agent3_report.get("burst_patterns"), 8),
            "copy_patterns": _safe_records(agent3_report.get("copy_patterns"), 8),
        }

    if strategy:
        context["agent4_strategie"] = {
            "markdown": _shorten(strategy.get("markdown", ""), 4200),
            "priority": strategy.get("priority"),
            "posture": strategy.get("posture"),
            "messages_prioritaires": strategy.get("messages_prioritaires", []),
            "plan_action": strategy.get("plan_action", []),
            "risques": strategy.get("risques", {}),
            "avoid": strategy.get("avoid", []),
        }

    if draft_pack:
        context["agent5_redaction"] = {
            "markdown": _shorten(draft_pack.get("markdown", ""), 5200),
            "messages": draft_pack.get("messages", {}),
            "guardrail": _safe_records(draft_pack.get("guardrail"), 12),
        }

    return context


def context_to_json(context: Dict[str, Any]) -> str:
    """Contexte JSON compact transmis au LLM."""
    return json.dumps(context, ensure_ascii=False, indent=2, default=str)


def _contains_any(q: str, words: List[str]) -> bool:
    return any(w in q for w in words)


@dataclass
class AgentChatbotCrise:
    """Chatbot qui répond à partir des résultats des autres agents."""

    def answer_deterministic(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        q = re.sub(r"\s+", " ", str(question or "").lower().strip())
        if not q:
            return {
                "answer": "Pose une question sur le diagnostic, les narratifs, la propagation, la stratégie ou les messages.",
                "sources": [],
                "mode": "déterministe",
            }

        a1 = context.get("agent1_diagnostic", {})
        a2 = context.get("agent2_narratifs", {})
        a3 = context.get("agent3_propagation", {})
        a4 = context.get("agent4_strategie", {})
        a5 = context.get("agent5_redaction", {})

        sources = []
        sections = []

        if _contains_any(q, ["diagnostic", "score", "kpi", "chiffre", "pic", "volume", "jour", "période", "periode", "retweet", "rt", "reach", "sentiment"]):
            sources.append("Agent 1 — Diagnostic")
            k = a1.get("kpis", {})
            sections.append(
                "### Réponse basée sur l'Agent 1 — Diagnostic\n"
                f"- Période analysée : **{a1.get('periode', {}).get('start', 'n/a')} → {a1.get('periode', {}).get('end', 'n/a')}**.\n"
                f"- Niveau : **{a1.get('niveau', 'n/a')}** ; Crisis Velocity Score : **{a1.get('score', 'n/a')}/100**.\n"
                f"- Messages : **{format_int(k.get('messages', 0))}** ; auteurs uniques : **{format_int(k.get('auteurs_uniques', 0))}**.\n"
                f"- Retweets/reposts : **{float(k.get('retweets_pct', 0) or 0):.1f}%** ; sentiment négatif : **{float(k.get('sentiment_negatif_pct', 0) or 0):.1f}%**.\n"
                f"- Narratif dominant détecté : **{a1.get('dominant_narrative', 'non identifié')}**.\n"
            )

        if _contains_any(q, ["narratif", "narratifs", "thème", "theme", "mutation", "acteur", "acteurs", "risque narratif", "angle"]):
            sources.append("Agent 2 — Narratifs / acteurs")
            risk = a2.get("risk_matrix", [])[:5]
            lines = ["### Réponse basée sur l'Agent 2 — Narratifs"]
            if risk:
                lines.append("Narratifs prioritaires détectés :")
                for r in risk:
                    lines.append(
                        f"- **{r.get('narrative', 'n/a')}** : {r.get('messages', 'n/a')} messages, "
                        f"score stratégique {r.get('strategic_risk_score', 'n/a')}/100, angle : {r.get('angle_de_reponse', 'n/a')}"
                    )
            else:
                lines.append(a2.get("markdown", "Agent 2 non lancé ou pas encore disponible."))
            sections.append("\n".join(lines))

        if _contains_any(q, ["propagation", "coordination", "bot", "bots", "copier", "copie", "gabarit", "vitesse", "cascade", "concentration", "amplification"]):
            sources.append("Agent 3 — Propagation / coordination prudente")
            coord = a3.get("coordination_score", {}) or {}
            prop = a3.get("propagation_metrics", {}) or {}
            sections.append(
                "### Réponse basée sur l'Agent 3 — Propagation\n"
                f"- Score de coordination prudente : **{coord.get('score', 'n/a')}/100** ({coord.get('label', 'signal à interpréter prudemment')}).\n"
                f"- Lecture : il s'agit de **signaux de propagation**, pas d'une preuve de bots ou de manipulation.\n"
                f"- Métriques disponibles : {json.dumps(prop, ensure_ascii=False, default=str)[:900]}.\n"
                f"\n{_shorten(a3.get('markdown', ''), 1600)}"
            )

        if _contains_any(q, ["stratégie", "strategie", "réponse", "reponse", "riposte", "quoi faire", "priorité", "priorite", "plan", "silence", "sur-réaction", "surreaction"]):
            sources.append("Agent 4 — Stratégie")
            sections.append(
                "### Réponse basée sur l'Agent 4 — Stratégie de riposte\n"
                f"- Priorité : **{a4.get('priority', 'n/a')}**.\n"
                f"- Posture recommandée : **{a4.get('posture', 'n/a')}**.\n"
                f"\n{_shorten(a4.get('markdown', 'Agent 4 non lancé ou pas encore disponible.'), 2200)}"
            )

        if _contains_any(q, ["message", "tweet", "post", "communiqué", "communique", "faq", "journaliste", "interne", "rédige", "redige", "formulation", "publier"]):
            sources.append("Agent 5 — Rédaction / garde-fou")
            sections.append(
                "### Réponse basée sur l'Agent 5 — Messages et garde-fous\n"
                f"{_shorten(a5.get('markdown', 'Agent 5 non lancé ou pas encore disponible.'), 2600)}"
            )

        if not sections:
            sources = ["Agents 1 à 5 — synthèse"]
            sections.append(
                "### Synthèse basée sur les Agents 1 à 5\n"
                f"Le diagnostic Agent 1 indique un niveau **{a1.get('niveau', 'n/a')}** avec un score de **{a1.get('score', 'n/a')}/100**. "
                f"Le narratif dominant est **{a1.get('dominant_narrative', 'non identifié')}**. "
                "Pour une question précise, demande par exemple : `Pourquoi le pic est important ?`, "
                "`Quels narratifs dominent ?`, `Y a-t-il des signaux de coordination ?`, "
                "`Quelle stratégie recommandes-tu ?` ou `Rédige un post X`."
            )

        answer = "\n\n".join(sections)
        answer += "\n\n---\n**Sources utilisées en priorité :** " + ", ".join(dict.fromkeys(sources)) + "."
        return {"answer": answer, "sources": list(dict.fromkeys(sources)), "mode": "déterministe"}


__all__ = ["AgentChatbotCrise", "build_agent6_context", "context_to_json"]
