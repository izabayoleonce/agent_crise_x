# -*- coding: utf-8 -*-
"""
CrisisAI War Room — Agents 2 et 3
=================================
Agent 2 : Cartographe des narratifs et acteurs
Agent 3 : Analyste de propagation et coordination prudente

Ces agents s'appuient sur le diagnostic Agent 1. Ils ne publient rien et ne posent
pas d'accusation : ils produisent des signaux, scores et éléments de lecture pour
alimenter l'Agent 4 puis l'Agent 5.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .agent1_core import format_int, pct


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _period_data(result: Dict[str, Any]) -> pd.DataFrame:
    data = result.get("period_data")
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return pd.DataFrame()


def _to_records(df: Any, limit: int = 10) -> List[Dict[str, Any]]:
    if not isinstance(df, pd.DataFrame) or len(df) == 0:
        return []
    small = df.head(limit).copy()
    for c in small.columns:
        if pd.api.types.is_datetime64_any_dtype(small[c]):
            small[c] = small[c].astype(str)
    return small.to_dict(orient="records")


def _safe_sum(data: pd.DataFrame, col: str) -> float:
    if col not in data.columns:
        return 0.0
    return float(pd.to_numeric(data[col], errors="coerce").fillna(0).sum())


def _safe_mean(data: pd.DataFrame, col: str) -> float:
    if col not in data.columns or len(data) == 0:
        return 0.0
    return float(pd.to_numeric(data[col], errors="coerce").fillna(0).mean())


def _neg_pct(data: pd.DataFrame) -> float:
    if "sentiment" not in data.columns or len(data) == 0:
        return 0.0
    return pct(data["sentiment"].astype(str).str.lower().isin(["negative", "negatif", "négatif", "neg"]).sum(), len(data))


def _rt_pct(data: pd.DataFrame) -> float:
    if "is_retweet" not in data.columns or len(data) == 0:
        return 0.0
    return pct(data["is_retweet"].sum(), len(data))


def _narrative_explode(data: pd.DataFrame) -> pd.DataFrame:
    if len(data) == 0 or "narratives" not in data.columns:
        return pd.DataFrame(columns=list(data.columns) + ["narrative"])
    tmp = data.copy()
    tmp["narrative"] = tmp["narratives"].apply(lambda x: x if isinstance(x, list) else [str(x)])
    return tmp.explode("narrative")


def risk_label(score: float) -> str:
    if score >= 80:
        return "Critique"
    if score >= 60:
        return "Élevé"
    if score >= 40:
        return "Modéré"
    return "Faible"


def _normalize_text_for_copy(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+", " URL ", text)
    text = re.sub(r"@\w+", " @user ", text)
    text = re.sub(r"#\w+", " #tag ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:220]


# -----------------------------------------------------------------------------
# Agent 2 — Narratifs / acteurs / risque
# -----------------------------------------------------------------------------


@dataclass
class AgentNarratifsActeurs:
    """Analyse les narratifs, leur risque et les acteurs qui les portent."""

    def analyze(self, result: Dict[str, Any], time_bucket: str = "hour", top_n: int = 10) -> Dict[str, Any]:
        data = _period_data(result)
        if len(data) == 0:
            raise ValueError("Agent 2 a besoin d'un résultat Agent 1 contenant period_data.")

        narrative_matrix = self.narrative_matrix(data)
        narrative_timeline = self.narrative_timeline(data, bucket=time_bucket)
        actor_narratives = self.actor_narratives(data, top_n=top_n)
        risk_matrix = self.narrative_risk_matrix(narrative_matrix)
        mutation = self.detect_mutation(narrative_timeline)
        samples = self.sample_posts_by_narrative(data, top_n=3)
        recommendations = self.narrative_recommendations(risk_matrix)

        report = {
            "agent": "Agent 2 — Cartographe des narratifs et acteurs",
            "period": result.get("periode", {}),
            "narrative_matrix": narrative_matrix,
            "narrative_timeline": narrative_timeline,
            "actor_narratives": actor_narratives,
            "risk_matrix": risk_matrix,
            "mutation_signals": mutation,
            "sample_posts_by_narrative": samples,
            "recommendations": recommendations,
        }
        report["markdown"] = self.to_markdown(report)
        return report

    def narrative_matrix(self, data: pd.DataFrame) -> pd.DataFrame:
        exp = _narrative_explode(data)
        if len(exp) == 0:
            return pd.DataFrame(columns=["narrative", "messages", "pct_messages", "reach", "shares", "negatif_pct", "rt_pct", "risk_mean", "strategic_risk_score"])
        total = len(data)
        rows = []
        for narrative, g in exp.groupby("narrative"):
            reach = _safe_sum(g, "reach")
            shares = _safe_sum(g, "shares")
            messages = len(g)
            neg = _neg_pct(g)
            rt = _rt_pct(g)
            risk_mean = _safe_mean(g, "risk_level")
            # Score 0-100 : poids volume + engagement + négativité + risque
            volume_component = min(25, pct(messages, total) * 0.45)
            reach_component = min(20, math.log1p(reach) / 18 * 20) if reach > 0 else 0
            share_component = min(15, math.log1p(shares) / 10 * 15) if shares > 0 else 0
            negativity_component = min(20, neg * 0.20)
            risk_component = min(20, (risk_mean / 5) * 20)
            score = round(volume_component + reach_component + share_component + negativity_component + risk_component, 1)
            rows.append({
                "narrative": narrative,
                "messages": int(messages),
                "pct_messages": round(pct(messages, total), 1),
                "reach": int(reach),
                "shares": int(shares),
                "negatif_pct": round(neg, 1),
                "rt_pct": round(rt, 1),
                "risk_mean": round(risk_mean, 2),
                "strategic_risk_score": score,
                "risk_label": risk_label(score),
            })
        return pd.DataFrame(rows).sort_values(["strategic_risk_score", "messages"], ascending=False).reset_index(drop=True)

    def narrative_timeline(self, data: pd.DataFrame, bucket: str = "hour") -> pd.DataFrame:
        bucket_col = "hour" if bucket == "hour" and "hour" in data.columns else "day"
        exp = _narrative_explode(data)
        if len(exp) == 0:
            return pd.DataFrame(columns=[bucket_col, "narrative", "messages", "reach", "risk_mean"])
        tl = exp.groupby([bucket_col, "narrative"]).agg(
            messages=("author", "size"),
            reach=("reach", "sum"),
            retweets=("is_retweet", "sum"),
            risk_mean=("risk_level", "mean"),
        ).reset_index().sort_values([bucket_col, "messages"], ascending=[True, False])
        tl["rt_pct"] = np.where(tl["messages"] > 0, tl["retweets"] / tl["messages"] * 100, 0)
        return tl

    def actor_narratives(self, data: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        exp = _narrative_explode(data)
        if len(exp) == 0:
            return pd.DataFrame()
        tab = exp.groupby(["author", "narrative"]).agg(
            messages=("author", "size"),
            reach=("reach", "sum"),
            shares=("shares", "sum"),
            likes=("likes", "sum"),
            risk_mean=("risk_level", "mean"),
        ).reset_index()
        tab["actor_narrative_score"] = (
            np.log1p(tab["reach"]) * 0.35 +
            np.log1p(tab["shares"]) * 0.30 +
            np.log1p(tab["likes"]) * 0.20 +
            np.log1p(tab["messages"]) * 0.15
        )
        return tab.sort_values("actor_narrative_score", ascending=False).head(top_n).reset_index(drop=True)

    def narrative_risk_matrix(self, narrative_matrix: pd.DataFrame) -> pd.DataFrame:
        if len(narrative_matrix) == 0:
            return narrative_matrix
        tab = narrative_matrix.copy()
        def action(row):
            n = str(row["narrative"]).lower()
            if "harc" in n or "menace" in n:
                return "Réponse sécurité + condamnation des menaces"
            if "argent" in n or "subvention" in n or "copinage" in n or "favoritisme" in n:
                return "Transparence sur critères, gouvernance et garde-fous"
            if "censure" in n or "idéologie" in n or "ideologie" in n:
                return "Neutralité institutionnelle, éviter le débat partisan"
            if "cnc" in n or "institution" in n:
                return "Clarification factuelle et preuve de cohérence institutionnelle"
            return "Monitoring, réponse seulement si le narratif monte"
        tab["angle_de_reponse"] = tab.apply(action, axis=1)
        return tab

    def detect_mutation(self, narrative_timeline: pd.DataFrame) -> pd.DataFrame:
        if len(narrative_timeline) == 0:
            return pd.DataFrame(columns=["period", "dominant_narrative", "messages", "previous", "mutation"])
        time_col = "hour" if "hour" in narrative_timeline.columns else "day"
        dom = narrative_timeline.sort_values([time_col, "messages"], ascending=[True, False]).groupby(time_col).head(1).copy()
        dom["previous"] = dom["narrative"].shift(1)
        dom["mutation"] = np.where(dom["previous"].notna() & (dom["previous"] != dom["narrative"]), "changement de narratif dominant", "stable")
        dom = dom.rename(columns={time_col: "period", "narrative": "dominant_narrative"})
        return dom[["period", "dominant_narrative", "messages", "previous", "mutation"]].reset_index(drop=True)

    def sample_posts_by_narrative(self, data: pd.DataFrame, top_n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        exp = _narrative_explode(data)
        samples: Dict[str, List[Dict[str, Any]]] = {}
        if len(exp) == 0:
            return samples
        for narrative, g in exp.groupby("narrative"):
            cols = [c for c in ["dt", "author", "engagement_type", "sentiment", "likes", "shares", "reach", "risk_level", "text_raw"] if c in g.columns]
            small = g.sort_values(["engagement_total", "reach", "shares"], ascending=False)[cols].head(top_n).copy()
            if "dt" in small.columns:
                small["dt"] = small["dt"].astype(str)
            samples[narrative] = small.to_dict(orient="records")
        return samples

    def narrative_recommendations(self, risk_matrix: pd.DataFrame) -> List[str]:
        if len(risk_matrix) == 0:
            return ["Continuer le monitoring : aucun narratif dominant exploitable."]
        recs = []
        for _, row in risk_matrix.head(4).iterrows():
            recs.append(f"{row['narrative']} : {row['angle_de_reponse']}.")
        recs.append("Ne pas répondre à tous les messages : prioriser les narratifs qui attaquent la confiance institutionnelle.")
        return recs

    def to_markdown(self, report: Dict[str, Any]) -> str:
        matrix = report.get("risk_matrix", pd.DataFrame())
        actors = report.get("actor_narratives", pd.DataFrame())
        mutation = report.get("mutation_signals", pd.DataFrame())
        lines: List[str] = []
        lines.append("## Agent 2 — Cartographie des narratifs et acteurs")
        lines.append("")
        if len(matrix):
            top = matrix.iloc[0]
            lines.append(f"**Narratif prioritaire :** {top['narrative']} — risque {top['risk_label']} ({top['strategic_risk_score']}/100).")
            lines.append("")
            lines.append("### Matrice narratifs")
            for _, row in matrix.head(6).iterrows():
                lines.append(f"- **{row['narrative']}** : {format_int(row['messages'])} occurrences, {row['pct_messages']}% des messages, risque {row['risk_label']} — {row['angle_de_reponse']}")
        if len(actors):
            lines.append("")
            lines.append("### Acteurs par narratif")
            for _, row in actors.head(6).iterrows():
                lines.append(f"- @{row['author']} porte surtout **{row['narrative']}** : {format_int(row['messages'])} messages, reach {format_int(row['reach'])}.")
        if len(mutation):
            changes = mutation[mutation["mutation"] != "stable"].head(5)
            lines.append("")
            lines.append("### Mutation narrative")
            if len(changes):
                for _, row in changes.iterrows():
                    lines.append(f"- {row['period']} : passage de **{row['previous']}** vers **{row['dominant_narrative']}**.")
            else:
                lines.append("- Pas de changement majeur du narratif dominant sur la période sélectionnée.")
        lines.append("")
        lines.append("### Recommandations Agent 2")
        for rec in report.get("recommendations", []):
            lines.append(f"- {rec}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Agent 3 — Propagation / coordination prudente
# -----------------------------------------------------------------------------


@dataclass
class AgentPropagationCoordination:
    """Analyse la vitesse, la concentration et les signaux faibles de coordination."""

    def analyze(self, result: Dict[str, Any], top_n: int = 10) -> Dict[str, Any]:
        data = _period_data(result)
        if len(data) == 0:
            raise ValueError("Agent 3 a besoin d'un résultat Agent 1 contenant period_data.")

        propagation = self.propagation_metrics(data)
        concentration = self.concentration_metrics(data, top_n=top_n)
        copy_patterns = self.detect_copy_patterns(data, min_count=3)
        burst_patterns = self.detect_bursts(data)
        coordination = self.coordination_score(propagation, concentration, copy_patterns, burst_patterns)
        action = self.operational_reading(coordination, propagation)

        report = {
            "agent": "Agent 3 — Propagation et coordination prudente",
            "period": result.get("periode", {}),
            "propagation_metrics": propagation,
            "concentration_metrics": concentration,
            "copy_patterns": copy_patterns,
            "burst_patterns": burst_patterns,
            "coordination_score": coordination,
            "operational_reading": action,
        }
        report["markdown"] = self.to_markdown(report)
        return report

    def propagation_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        start = data["dt"].min()
        end = data["dt"].max()
        hours = max(1.0, (end - start).total_seconds() / 3600)
        messages = len(data)
        hourly = data.groupby("hour").agg(messages=("author", "size"), auteurs=("author", "nunique"), retweets=("is_retweet", "sum"), reach=("reach", "sum")).reset_index() if "hour" in data.columns else pd.DataFrame()
        peak_hour_messages = int(hourly["messages"].max()) if len(hourly) else messages
        avg_hour_messages = float(messages / hours)
        acceleration_ratio = round(peak_hour_messages / max(avg_hour_messages, 1e-9), 2)
        return {
            "start": str(start),
            "end": str(end),
            "duration_hours": round(hours, 2),
            "messages": int(messages),
            "messages_per_hour": round(avg_hour_messages, 2),
            "peak_hour_messages": peak_hour_messages,
            "acceleration_ratio": acceleration_ratio,
            "retweet_pct": round(_rt_pct(data), 1),
            "unique_authors": int(data["author"].nunique()) if "author" in data.columns else 0,
            "reach": int(_safe_sum(data, "reach")),
            "impressions": int(_safe_sum(data, "impressions")),
        }

    def concentration_metrics(self, data: pd.DataFrame, top_n: int = 10) -> Dict[str, Any]:
        if len(data) == 0:
            return {}
        by_author = data.groupby("author").agg(
            messages=("author", "size"),
            reach=("reach", "sum"),
            shares=("shares", "sum"),
            likes=("likes", "sum"),
            retweets=("is_retweet", "sum"),
        ).reset_index().sort_values(["reach", "shares", "messages"], ascending=False)
        top = by_author.head(top_n).copy()
        total_messages = max(len(data), 1)
        total_reach = max(_safe_sum(data, "reach"), 1)
        total_shares = max(_safe_sum(data, "shares"), 1)
        return {
            "top_authors_table": top,
            "top10_message_share_pct": round(top["messages"].sum() / total_messages * 100, 1),
            "top10_reach_share_pct": round(top["reach"].sum() / total_reach * 100, 1),
            "top10_share_share_pct": round(top["shares"].sum() / total_shares * 100, 1),
            "authors_total": int(by_author["author"].nunique()),
        }

    def detect_copy_patterns(self, data: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
        if "text_raw" not in data.columns or len(data) == 0:
            return pd.DataFrame(columns=["fingerprint", "count", "authors", "first_seen", "last_seen", "example"])
        tmp = data.copy()
        tmp["fingerprint"] = tmp["text_raw"].map(_normalize_text_for_copy)
        grp = tmp.groupby("fingerprint").agg(
            count=("author", "size"),
            authors=("author", "nunique"),
            first_seen=("dt", "min"),
            last_seen=("dt", "max"),
            reach=("reach", "sum"),
            example=("text_raw", "first"),
        ).reset_index()
        grp = grp[(grp["count"] >= min_count) & (grp["fingerprint"].str.len() > 40)].copy()
        if len(grp) == 0:
            return grp
        grp["first_seen"] = grp["first_seen"].astype(str)
        grp["last_seen"] = grp["last_seen"].astype(str)
        grp["example"] = grp["example"].astype(str).str.slice(0, 260)
        return grp.sort_values(["count", "authors", "reach"], ascending=False).head(15).reset_index(drop=True)

    def detect_bursts(self, data: pd.DataFrame) -> pd.DataFrame:
        if "hour" not in data.columns or len(data) == 0:
            return pd.DataFrame(columns=["hour", "messages", "authors", "retweet_pct", "burst_level"])
        tl = data.groupby("hour").agg(
            messages=("author", "size"),
            authors=("author", "nunique"),
            retweets=("is_retweet", "sum"),
            reach=("reach", "sum"),
        ).reset_index().sort_values("hour")
        mean = tl["messages"].mean()
        std = tl["messages"].std(ddof=0) or 1
        tl["zscore"] = (tl["messages"] - mean) / std
        tl["retweet_pct"] = np.where(tl["messages"] > 0, tl["retweets"] / tl["messages"] * 100, 0)
        tl["burst_level"] = np.where(tl["zscore"] >= 3, "pic majeur", np.where(tl["zscore"] >= 2, "pic notable", "normal"))
        return tl.sort_values(["zscore", "messages"], ascending=False).head(12).reset_index(drop=True)

    def coordination_score(self, propagation: Dict[str, Any], concentration: Dict[str, Any], copy_patterns: pd.DataFrame, burst_patterns: pd.DataFrame) -> Dict[str, Any]:
        rt_component = min(25, float(propagation.get("retweet_pct", 0)) * 0.25)
        concentration_component = min(25, float(concentration.get("top10_reach_share_pct", 0)) * 0.35)
        copy_component = min(25, len(copy_patterns) * 3.5)
        burst_component = min(25, len(burst_patterns[burst_patterns.get("burst_level", "") != "normal"]) * 4 if isinstance(burst_patterns, pd.DataFrame) and len(burst_patterns) else 0)
        score = round(rt_component + concentration_component + copy_component + burst_component, 1)
        if score >= 75:
            label = "Signaux forts d'amplification structurée"
        elif score >= 50:
            label = "Signaux modérés d'amplification synchronisée"
        elif score >= 25:
            label = "Quelques signaux faibles à surveiller"
        else:
            label = "Propagation surtout organique ou non démontrée"
        return {
            "score": min(100, score),
            "label": label,
            "rt_component": round(rt_component, 1),
            "concentration_component": round(concentration_component, 1),
            "copy_component": round(copy_component, 1),
            "burst_component": round(burst_component, 1),
            "prudence": "Ce score ne prouve pas l'existence de bots. Il mesure des signaux de concentration, synchronie et copier-coller à vérifier humainement.",
        }

    def operational_reading(self, coordination: Dict[str, Any], propagation: Dict[str, Any]) -> List[str]:
        recs = []
        if propagation.get("retweet_pct", 0) >= 70:
            recs.append("La crise est portée par amplification : surveiller les sources reprises plutôt que répondre à tous les retweets.")
        if propagation.get("acceleration_ratio", 0) >= 4:
            recs.append("La vitesse de pic est forte : préparer une réponse courte et une FAQ avant le prochain rebond.")
        if coordination.get("score", 0) >= 50:
            recs.append("Formuler 'signaux d'amplification synchronisée' et éviter d'accuser des bots sans preuve.")
        recs.append("Prioriser les publications sources, les relais médias et les comptes passerelles entre communautés.")
        return recs

    def to_markdown(self, report: Dict[str, Any]) -> str:
        prop = report.get("propagation_metrics", {})
        conc = report.get("concentration_metrics", {})
        coord = report.get("coordination_score", {})
        copies = report.get("copy_patterns", pd.DataFrame())
        bursts = report.get("burst_patterns", pd.DataFrame())
        lines: List[str] = []
        lines.append("## Agent 3 — Propagation et coordination prudente")
        lines.append("")
        lines.append(f"**Score de coordination prudente :** {coord.get('score', 0)}/100 — {coord.get('label', 'n/a')}.")
        lines.append(f"_Prudence : {coord.get('prudence', '')}_")
        lines.append("")
        lines.append("### Propagation")
        lines.append(f"- {format_int(prop.get('messages', 0))} messages sur {prop.get('duration_hours', 0)} h, soit {prop.get('messages_per_hour', 0)} messages/h en moyenne.")
        lines.append(f"- Pic horaire : {format_int(prop.get('peak_hour_messages', 0))} messages ; ratio d'accélération : ×{prop.get('acceleration_ratio', 0)}.")
        lines.append(f"- Retweets/reposts : {prop.get('retweet_pct', 0)}% ; auteurs uniques : {format_int(prop.get('unique_authors', 0))}.")
        lines.append("")
        lines.append("### Concentration")
        lines.append(f"- Top comptes : {conc.get('top10_message_share_pct', 0)}% des messages, {conc.get('top10_reach_share_pct', 0)}% du reach, {conc.get('top10_share_share_pct', 0)}% des shares.")
        if isinstance(copies, pd.DataFrame) and len(copies):
            lines.append("")
            lines.append("### Copier-coller / gabarits")
            for _, row in copies.head(5).iterrows():
                lines.append(f"- Gabarit repris {format_int(row['count'])} fois par {format_int(row['authors'])} auteurs ; exemple : {str(row['example'])[:140]}...")
        if isinstance(bursts, pd.DataFrame) and len(bursts):
            lines.append("")
            lines.append("### Pics horaires")
            for _, row in bursts.head(5).iterrows():
                lines.append(f"- {row['hour']} : {format_int(row['messages'])} messages, {row['retweet_pct']:.1f}% RT — {row['burst_level']}.")
        lines.append("")
        lines.append("### Lecture opérationnelle")
        for rec in report.get("operational_reading", []):
            lines.append(f"- {rec}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Exports Agents 2/3
# -----------------------------------------------------------------------------


def agent23_export_files(agent2_report: Optional[Dict[str, Any]] = None, agent3_report: Optional[Dict[str, Any]] = None) -> Dict[str, bytes]:
    files: Dict[str, bytes] = {}
    if agent2_report:
        files["agent2_narratifs_acteurs.md"] = agent2_report.get("markdown", "").encode("utf-8")
        for key, name in [
            ("narrative_matrix", "agent2_matrice_narratifs.csv"),
            ("narrative_timeline", "agent2_timeline_narratifs.csv"),
            ("actor_narratives", "agent2_acteurs_par_narratif.csv"),
            ("risk_matrix", "agent2_risk_matrix.csv"),
            ("mutation_signals", "agent2_mutations_narratives.csv"),
        ]:
            obj = agent2_report.get(key)
            if isinstance(obj, pd.DataFrame):
                files[name] = obj.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        safe_json = {k: v for k, v in agent2_report.items() if k not in {"narrative_matrix", "narrative_timeline", "actor_narratives", "risk_matrix", "mutation_signals"}}
        files["agent2_resume.json"] = json.dumps(safe_json, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    if agent3_report:
        files["agent3_propagation_coordination.md"] = agent3_report.get("markdown", "").encode("utf-8")
        for key, name in [
            ("copy_patterns", "agent3_copier_coller_gabarits.csv"),
            ("burst_patterns", "agent3_pics_horaires.csv"),
        ]:
            obj = agent3_report.get(key)
            if isinstance(obj, pd.DataFrame):
                files[name] = obj.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        conc = agent3_report.get("concentration_metrics", {})
        if isinstance(conc, dict) and isinstance(conc.get("top_authors_table"), pd.DataFrame):
            files["agent3_top_comptes_concentration.csv"] = conc["top_authors_table"].to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
        safe_json = {k: v for k, v in agent3_report.items() if k not in {"copy_patterns", "burst_patterns"}}
        if isinstance(safe_json.get("concentration_metrics"), dict):
            safe_json["concentration_metrics"] = {kk: vv for kk, vv in safe_json["concentration_metrics"].items() if kk != "top_authors_table"}
        files["agent3_resume.json"] = json.dumps(safe_json, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    return files


def make_zip(files: Dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    buffer.seek(0)
    return buffer.getvalue()
