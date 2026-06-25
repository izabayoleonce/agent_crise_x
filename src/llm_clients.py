# -*- coding: utf-8 -*-
"""Clients LLM optionnels OpenRouter pour CrisisAI War Room."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import requests

from .agent1_core import result_context_json
from .agent45_core import build_crisis_context


DEFAULT_MODEL = "openai/gpt-4o-mini"


def _openrouter_chat(
    *,
    api_key: Optional[str],
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1200,
    temperature: float = 0.2,
) -> str:
    key = api_key or os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY manquant.")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://streamlit.io/",
            "X-Title": "CrisisAI War Room",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_openrouter_brief(result: dict, api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> str:
    """Génère un brief Agent 1 rédigé via OpenRouter, sans inventer de chiffres."""
    context = result_context_json(result)
    system = (
        "Tu es un analyste senior en communication de crise et réseaux sociaux. "
        "Tu restes neutre, factuel et prudent. Tu analyses une dynamique virale, pas une fake news. "
        "Tu n'inventes aucun chiffre : tu utilises uniquement le JSON fourni. "
        "Tu écris en français professionnel, utile pour une cellule de crise."
    )
    user = (
        "À partir du contexte JSON suivant, rédige un brief en 6 parties : "
        "1) diagnostic, 2) narratifs, 3) acteurs, 4) propagation, 5) risques, "
        "6) recommandations immédiates.\n\n"
        f"CONTEXTE_JSON:\n{context}"
    )
    return _openrouter_chat(api_key=api_key, model=model, system=system, user=user, max_tokens=1200, temperature=0.2)


def generate_openrouter_strategy(
    result: Dict[str, Any],
    strategy: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Améliore la stratégie Agent 4 via OpenRouter, à partir de données vérifiées."""
    context = build_crisis_context(result)
    payload = {
        "context_agent1": context,
        "strategie_deterministe": {k: v for k, v in strategy.items() if k not in {"context", "markdown"}},
    }
    system = (
        "Tu es directeur de communication de crise pour une institution publique. "
        "Tu proposes une stratégie neutre, factuelle, prudente et validable par une cellule humaine. "
        "Tu n'accuses jamais de bots ou de manipulation sans preuve. "
        "Tu ne crées aucun chiffre : tu t'appuies uniquement sur le JSON fourni."
    )
    user = (
        "Rédige la stratégie finale de riposte en français professionnel. Structure : "
        "1) diagnostic, 2) objectif, 3) posture, 4) messages prioritaires, "
        "5) plan 0-24h, 6) risques si silence / sur-réaction, 7) garde-fous.\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )
    return _openrouter_chat(api_key=api_key, model=model, system=system, user=user, max_tokens=1500, temperature=0.2)


def generate_openrouter_drafts(
    result: Dict[str, Any],
    strategy: Dict[str, Any],
    draft_pack: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Améliore les messages Agent 5 via OpenRouter, avec garde-fous."""
    payload = {
        "context_agent1": build_crisis_context(result),
        "strategie_agent4": {k: v for k, v in strategy.items() if k not in {"context", "markdown"}},
        "messages_deterministes": draft_pack.get("messages", {}),
    }
    system = (
        "Tu es rédacteur senior en communication de crise institutionnelle. "
        "Tu rédiges des messages courts, sobres, factuels et apaisants. "
        "Tu évites les attaques personnelles, les formulations politiques, les accusations non prouvées et les promesses non validées. "
        "Aucun message ne doit dire qu'il est publié automatiquement : l'humain valide."
    )
    user = (
        "Réécris et améliore les messages pour les rendre professionnels. Structure obligatoire : "
        "1) Post X court, 2) Thread X, 3) Communiqué, 4) FAQ, 5) Réponse journaliste, "
        "6) Message interne, 7) Garde-fous avant publication.\n\n"
        f"JSON:\n{json.dumps(payload, ensure_ascii=False, indent=2, default=str)}"
    )
    return _openrouter_chat(api_key=api_key, model=model, system=system, user=user, max_tokens=1800, temperature=0.25)



def generate_openrouter_agent6_answer(
    question: str,
    agent6_context: Dict[str, Any],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> str:
    """Répond via OpenRouter à partir du contexte prioritaire des Agents 1 à 5."""
    system = (
        "Tu es l'Agent 6, chatbot d'une War Room IA de communication de crise. "
        "Tu réponds en français clair, professionnel et prudent. "
        "Tu bases ta réponse uniquement sur le CONTEXTE_JSON issu des Agents 1 à 5. "
        "Tu n'inventes aucun chiffre, aucune source et aucune preuve. "
        "Si une information n'est pas disponible dans le contexte, dis-le clairement. "
        "Les signaux de coordination ne doivent jamais être présentés comme une preuve de bots. "
        "Tu rappelles que l'IA assiste la cellule de crise et que l'humain valide les décisions."
    )
    user = (
        "Question utilisateur :\n"
        f"{question}\n\n"
        "Réponds avec cette structure :\n"
        "1) Réponse directe\n"
        "2) Éléments issus des agents utilisés\n"
        "3) Limites / prudence\n"
        "4) Action recommandée si utile\n\n"
        "CONTEXTE_JSON :\n"
        f"{json.dumps(agent6_context, ensure_ascii=False, indent=2, default=str)}"
    )
    return _openrouter_chat(api_key=api_key, model=model, system=system, user=user, max_tokens=1800, temperature=0.2)
