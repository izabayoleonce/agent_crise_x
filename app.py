# -*- coding: utf-8 -*-
"""
CrisisAI War Room — Agents 1, 2, 3, 4, 5, 6
Déploiement : streamlit run app.py
"""

from __future__ import annotations

from datetime import datetime, time
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agent1_core import (
    AnalysteDeCrise,
    auto_map_columns,
    build_daily_timeline,
    build_hourly_timeline,
    compute_kpis,
    crisis_level,
    detect_peaks_daily,
    detect_peaks_hourly,
    engagement_type_table,
    format_int,
    make_zip as make_agent1_zip,
    narratives_table,
    preprocess_dataset,
    read_dataset,
    result_to_export_files,
    sentiment_table,
    top_authors,
    top_hashtags,
)
from src.agent23_core import (
    AgentNarratifsActeurs,
    AgentPropagationCoordination,
    agent23_export_files,
)
from src.agent45_core import (
    AgentRedacteurGardeFou,
    AgentStrategieRiposte,
    agent45_export_files,
    build_crisis_context,
)
from src.agent6_core import (
    AgentChatbotCrise,
    build_agent6_context,
)
from src.llm_clients import (
    DEFAULT_MODEL,
    generate_openrouter_agent6_answer,
    generate_openrouter_brief,
    generate_openrouter_drafts,
    generate_openrouter_strategy,
)


def get_streamlit_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


st.set_page_config(
    page_title="CrisisAI War Room — Agents 1 à 6",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


CUSTOM_CSS = """
<style>
.block-container {padding-top: 1.9rem; padding-bottom: 2rem;}
.metric-card {
    border: 1px solid rgba(128,128,128,0.20);
    border-radius: 18px;
    padding: 1rem 1.1rem;
    background: rgba(127,127,127,0.055);
    min-height: 104px;
}
.metric-title {font-size: 0.85rem; color: #777; margin-bottom: 0.25rem;}
.metric-value {font-size: 1.75rem; font-weight: 800; line-height: 1.2;}
.metric-note {font-size: 0.8rem; color: #777; margin-top: 0.3rem;}
.big-title {font-size: 2.1rem; font-weight: 900;}
.subtitle {color: #777; margin-bottom: 1rem;}
.warning-box {border-left: 5px solid #ffb000; padding: .8rem 1rem; background: rgba(255,176,0,.08); border-radius: 10px;}
.success-box {border-left: 5px solid #1aaf5d; padding: .8rem 1rem; background: rgba(26,175,93,.08); border-radius: 10px;}
.agent-box {border: 1px solid rgba(128,128,128,0.18); border-radius: 16px; padding: 1rem; background: rgba(127,127,127,0.04);}
.small-muted {font-size: 0.85rem; color: #777;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Utils Streamlit
# -----------------------------------------------------------------------------


def show_metric_card(title: str, value: str, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def cached_preprocess(file_bytes: bytes, file_name: str, mapping: Optional[Dict[str, Optional[str]]] = None):
    class UploadedLike:
        def __init__(self, data: bytes, name: str):
            self._data = data
            self.name = name
        def getvalue(self):
            return self._data

    raw = read_dataset(UploadedLike(file_bytes, file_name))
    df, used_mapping = preprocess_dataset(raw, mapping=mapping)
    return raw, df, used_mapping


def plot_timeline_daily(daily: pd.DataFrame):
    fig = px.line(daily, x="day", y="messages", markers=True, title="Timeline journalière — volume de messages")
    fig.update_layout(height=420, xaxis_title="Jour", yaxis_title="Messages")
    return fig


def plot_timeline_hourly(hourly: pd.DataFrame):
    fig = px.line(hourly, x="hour", y="messages", markers=True, title="Timeline horaire")
    fig.update_layout(height=420, xaxis_title="Heure", yaxis_title="Messages")
    return fig


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str, orientation: str = "v"):
    if df is None or len(df) == 0:
        return go.Figure()
    if orientation == "h":
        plot_df = df.copy().iloc[::-1]
        fig = px.bar(plot_df, x=x, y=y, orientation="h", title=title)
    else:
        fig = px.bar(df, x=x, y=y, title=title)
    fig.update_layout(height=420)
    return fig


def plot_score(score: float, title: str = "Crisis Velocity Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={"suffix": "/100"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"thickness": 0.35},
            "steps": [
                {"range": [0, 40], "color": "rgba(0,180,90,0.18)"},
                {"range": [40, 60], "color": "rgba(255,200,0,0.20)"},
                {"range": [60, 80], "color": "rgba(255,120,0,0.22)"},
                {"range": [80, 100], "color": "rgba(255,0,0,0.20)"},
            ],
        },
        title={"text": title},
    ))
    fig.update_layout(height=330, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def mapping_editor(raw_df: pd.DataFrame, auto_mapping: Dict[str, Optional[str]]) -> Dict[str, Optional[str]]:
    st.markdown("### Mapping des colonnes")
    st.caption("L'app détecte les colonnes automatiquement. Corrige seulement si une colonne importante est mal reconnue.")
    columns = [None] + list(raw_df.columns)
    mapping = dict(auto_mapping)

    required = ["date", "author", "text"]
    optional = ["sentiment", "likes", "comments", "shares", "reach", "impressions", "engagement_type", "hashtags", "mentions", "followers", "text_norm", "repost_of"]

    cols = st.columns(3)
    for i, key in enumerate(required):
        current = mapping.get(key)
        idx = columns.index(current) if current in columns else 0
        mapping[key] = cols[i].selectbox(f"{key} *", columns, index=idx, key=f"map_{key}")

    with st.expander("Colonnes optionnelles", expanded=False):
        grid = st.columns(3)
        for i, key in enumerate(optional):
            current = mapping.get(key)
            idx = columns.index(current) if current in columns else 0
            mapping[key] = grid[i % 3].selectbox(key, columns, index=idx, key=f"map_{key}")
    return mapping


def ensure_active_result(agent: AnalysteDeCrise, peaks_d: pd.DataFrame, top_n: int):
    if "last_result" in st.session_state:
        return st.session_state["last_result"]
    if len(peaks_d):
        result = agent.analyze_peak_day(rank=1, top_n=top_n)
    else:
        result = agent.analyze_period(top_n=top_n)
    st.session_state["last_result"] = result
    return result


def ensure_agent2(result, top_n: int):
    if st.session_state.get("agent2_result_id") != id(result) or "agent2_report" not in st.session_state:
        st.session_state["agent2_report"] = AgentNarratifsActeurs().analyze(result, time_bucket="hour", top_n=top_n)
        st.session_state["agent2_result_id"] = id(result)
    return st.session_state["agent2_report"]


def ensure_agent3(result, top_n: int):
    if st.session_state.get("agent3_result_id") != id(result) or "agent3_report" not in st.session_state:
        st.session_state["agent3_report"] = AgentPropagationCoordination().analyze(result, top_n=top_n)
        st.session_state["agent3_result_id"] = id(result)
    return st.session_state["agent3_report"]


def get_api_key() -> str:
    return st.session_state.get("openrouter_key_input", "") or get_streamlit_secret("OPENROUTER_API_KEY", "")




def build_strategy_safe(strategist, result, **kwargs):
    """
    Compatibilité entre deux versions de src/agent45_core.py.

    - Nouvelle version : build_strategy(..., agent2_report=..., agent3_report=...)
    - Ancienne version : build_strategy(...) sans agent2_report / agent3_report

    Cette fonction évite le crash Streamlit si app.py et agent45_core.py ne sont
    pas parfaitement synchronisés.
    """
    try:
        return strategist.build_strategy(result, **kwargs)
    except TypeError as e:
        msg = str(e)
        if "agent2_report" in msg or "agent3_report" in msg or "unexpected keyword argument" in msg:
            kwargs.pop("agent2_report", None)
            kwargs.pop("agent3_report", None)
            return strategist.build_strategy(result, **kwargs)
        raise

@st.fragment
def interface_chatbot_agent6(agent6_context, use_openrouter, openrouter_model):
    """Ce fragment gère le chat sans recharger toute l'application."""
    
    with st.expander("Questions rapides (Copiez-collez la question de votre choix)", expanded=True):
        qcols = st.columns(3)
        quick_questions = [
            "Pourquoi le pic principal est-il important ?",
            "Quels narratifs dominent et lesquels sont les plus risqués ?",
            "Y a-t-il des signaux de coordination ou de copier-coller ?",
            "Quelle stratégie de réponse recommandes-tu ?",
            "Rédige un post X prudent pour le CNC.",
            "Quelles sont les limites de notre analyse ?",
        ]
        for i, qq in enumerate(quick_questions):
            with qcols[i % 3]:
                st.code(qq, language="plaintext")

    if "agent6_history" not in st.session_state:
        st.session_state["agent6_history"] = []

    question = st.text_area(
        "Ta question à l'Agent 6",
        value="",
        placeholder="Exemple : Explique-moi pourquoi l'angle argent public est dangereux pour le CNC.",
        height=90,
    )

    col_send, col_clear = st.columns([1, 1])
    ask = col_send.button("💬 Poser la question", type="primary")
    clear = col_clear.button("🧹 Vider la conversation")
    
    if clear:
        st.session_state["agent6_history"] = []
        st.rerun()

    # Traitement de la question
    if ask and question.strip():
        bot = AgentChatbotCrise()
        mode = "déterministe"
        try:
            if use_openrouter:
                api_key = get_api_key()
                with st.spinner("Agent 6 interroge OpenRouter..."):
                    answer = generate_openrouter_agent6_answer(
                        question,
                        agent6_context,
                        api_key=api_key,
                        model=openrouter_model,
                    )
                mode = "OpenRouter"
                sources = ["Agents 1 à 5 via contexte JSON"]
            else:
                pack = bot.answer_deterministic(question, agent6_context)
                answer = pack["answer"]
                sources = pack.get("sources", [])
        except Exception as e:
            st.warning(f"Erreur LLM. Réponse déterministe utilisée. Détail : {e}")
            pack = bot.answer_deterministic(question, agent6_context)
            answer = pack["answer"]
            sources = pack.get("sources", [])
            mode = "déterministe fallback"

        st.session_state["agent6_history"].append({
            "question": question.strip(),
            "answer": answer,
            "mode": mode,
            "sources": sources,
        })

    # Affichage de l'historique
    st.markdown("### Conversation")
    if not st.session_state["agent6_history"]:
        st.caption("Aucune question posée pour l'instant.")
    
    for i, turn in enumerate(st.session_state["agent6_history"], start=1):
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.caption(f"Mode : {turn.get('mode', 'n/a')} | Sources : {', '.join(turn.get('sources', []))}")
            st.markdown(turn["answer"])

    # Bouton de téléchargement
    if st.session_state["agent6_history"]:
        conv_md = "\n\n".join(
            [f"## Question {i}\n{t['question']}\n\n### Réponse\n{t['answer']}" for i, t in enumerate(st.session_state["agent6_history"], start=1)]
        )
        st.download_button(
            "⬇️ Télécharger la conversation Agent 6 (.md)",
            data=conv_md.encode("utf-8"),
            file_name="conversation_agent6_chatbot.md",
            mime="text/markdown",
        )
# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

st.sidebar.title("🚨 CrisisAI War Room")
st.sidebar.caption("Agents 1 à 6 — Diagnostic → Narratifs → Propagation → Stratégie → Messages → Chatbot")

uploaded = st.sidebar.file_uploader(
    "Importer le corpus X/Twitter",
    type=["csv", "xlsx", "xls"],
    help="Le fichier doit contenir au minimum une date, un auteur et un texte/message.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Réglages")
top_n = st.sidebar.slider("Nombre d'éléments dans les tops", 5, 30, 10)
show_mapping = st.sidebar.checkbox("Afficher/corriger le mapping", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("LLM optionnel")
use_openrouter = st.sidebar.checkbox("Activer OpenRouter pour réécriture pro", value=False)
openrouter_model = st.sidebar.text_input("Modèle OpenRouter", value=DEFAULT_MODEL)
st.session_state["openrouter_key_input"] = st.sidebar.text_input("Clé OpenRouter temporaire", type="password", value="")
st.sidebar.caption("Tu peux aussi définir OPENROUTER_API_KEY dans les secrets Streamlit. Ne mets jamais la clé sur GitHub.")


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.markdown('<div class="big-title">CrisisAI War Room — Agents 1 à 6</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Salle de crise IA complète : diagnostic, narratifs, propagation, stratégie, messages, garde-fous et chatbot.</div>',
    unsafe_allow_html=True,
)

if uploaded is None:
    st.markdown(
        """
        <div class="warning-box">
        <b>Commence ici :</b> importe ton fichier <code>data.xlsx</code> ou un CSV équivalent dans la barre latérale.
        L'application va produire le pipeline complet : Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5 → Agent 6.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("Colonnes minimales attendues : date, auteur, texte. Les colonnes likes, shares, reach, sentiment, hashtags, type d'engagement sont utilisées si elles existent.")
    st.stop()

# Lecture raw pour mapping simple
file_bytes = uploaded.getvalue()
raw_preview = read_dataset(uploaded)
auto_mapping = auto_map_columns(raw_preview)

if show_mapping:
    mapping = mapping_editor(raw_preview, auto_mapping)
else:
    mapping = auto_mapping

try:
    with st.spinner("Nettoyage et enrichissement du corpus..."):
        raw_df, df, used_mapping = cached_preprocess(file_bytes, uploaded.name, mapping)
except Exception as e:
    st.error(f"Impossible de préparer le corpus : {e}")
    with st.expander("Voir les colonnes détectées"):
        st.write(list(raw_preview.columns))
        st.json(auto_mapping)
    st.stop()

agent = AnalysteDeCrise(df)
daily = build_daily_timeline(df)
hourly = build_hourly_timeline(df)
peaks_d = detect_peaks_daily(df, n=10)
peaks_h = detect_peaks_hourly(df, n=20)
kpis = compute_kpis(df)

# -----------------------------------------------------------------------------
# KPIs globaux
# -----------------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
with c1:
    show_metric_card("Messages", format_int(kpis["messages"]), f"{format_int(kpis['auteurs_uniques'])} auteurs uniques")
with c2:
    show_metric_card("Retweets/Reposts", f"{kpis['retweets_pct']:.1f}%", f"{format_int(kpis['retweets'])} messages")
with c3:
    show_metric_card("Reach cumulé", format_int(kpis["reach"]), f"Impressions : {format_int(kpis['impressions'])}")
with c4:
    show_metric_card("Sentiment négatif", f"{kpis['sentiment_negatif_pct']:.1f}%", f"Risque moyen : {kpis['risk_moyen']:.2f}/5")

st.caption(f"Période couverte : {kpis['debut']} → {kpis['fin']}")

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
    "📊 Vue globale",
    "🔥 Pics détectés",
    "🤖 Agent 1 — Diagnostic",
    "🧠 Agent 2 — Narratifs",
    "🌐 Agent 3 — Propagation",
    "🧭 Agent 4 — Stratégie",
    "✍️ Agent 5 — Rédaction",
    "🔗 Orchestration Top 1",
    "💬 Agent 6 — Chatbot",
    "📦 Exports",
])

with tab1:
    st.subheader("Vue globale du corpus")
    st.plotly_chart(plot_timeline_daily(daily), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_bar(engagement_type_table(df), x="type", y="count", title="Types d'engagement"), use_container_width=True)
    with right:
        st.plotly_chart(plot_bar(sentiment_table(df), x="sentiment", y="count", title="Sentiment"), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(plot_bar(narratives_table(df).head(10), x="count", y="narrative", title="Narratifs dominants", orientation="h"), use_container_width=True)
    with right:
        hashtags = top_hashtags(df, n=10)
        if len(hashtags):
            st.plotly_chart(plot_bar(hashtags, x="count", y="hashtag", title="Top hashtags", orientation="h"), use_container_width=True)
        else:
            st.info("Aucun hashtag détecté.")

    with st.expander("Aperçu du corpus enrichi"):
        cols = ["dt", "author", "engagement_type", "sentiment", "main_narrative", "risk_level", "text_raw"]
        st.dataframe(df[[c for c in cols if c in df.columns]].head(100), use_container_width=True)

with tab2:
    st.subheader("Pics automatiques")
    st.markdown("L'Agent 1 détecte les pics par volume et z-score. C'est la base pour choisir la période à analyser.")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Top pics journaliers")
        st.dataframe(peaks_d, use_container_width=True)
    with right:
        st.markdown("#### Top pics horaires")
        st.dataframe(peaks_h, use_container_width=True)

    best_days = list(peaks_d["day"].head(2)) if len(peaks_d) else []
    focus = df[df["day"].isin(best_days)].copy() if best_days else df.head(0)
    if len(focus):
        st.plotly_chart(plot_timeline_hourly(build_hourly_timeline(focus)), use_container_width=True)

with tab3:
    st.subheader("Agent 1 — Diagnostic d'une période")
    st.markdown("Choisis une période, puis l'agent produit un brief, des tops et des tableaux exploitables par les autres agents.")

    min_dt, max_dt = df["dt"].min(), df["dt"].max()
    default_start = peaks_d.iloc[0]["day"].to_pydatetime() if len(peaks_d) else min_dt.to_pydatetime()
    default_end = default_start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    default_end = min(default_end, max_dt.to_pydatetime())

    col_a, col_b, col_c = st.columns([1, 1, 1])
    with col_a:
        start_date = st.date_input("Date début", value=default_start.date(), min_value=min_dt.date(), max_value=max_dt.date())
        start_time = st.time_input("Heure début", value=time(0, 0))
    with col_b:
        end_date = st.date_input("Date fin", value=default_end.date(), min_value=min_dt.date(), max_value=max_dt.date())
        end_time = st.time_input("Heure fin", value=time(23, 59))
    with col_c:
        st.markdown("#### Raccourci")
        peak_rank = st.selectbox("Analyser le pic journalier n°", list(range(1, min(10, len(peaks_d)) + 1)), index=0) if len(peaks_d) else 1
        use_peak = st.button("Utiliser ce pic")

    if use_peak and len(peaks_d):
        result = agent.analyze_peak_day(rank=int(peak_rank), top_n=top_n)
    else:
        start = datetime.combine(start_date, start_time)
        end = datetime.combine(end_date, end_time)
        try:
            result = agent.analyze_period(start=start, end=end, top_n=top_n)
        except Exception as e:
            st.error(str(e))
            st.stop()

    st.session_state["last_result"] = result
    # Dès qu'on change le diagnostic, on invalide les agents suivants.
    st.session_state.pop("agent2_report", None)
    st.session_state.pop("agent3_report", None)
    st.session_state.pop("agent4_strategy", None)
    st.session_state.pop("agent5_drafts", None)
    st.session_state.pop("agent6_history", None)

    score = result["crisis_velocity_score"]
    score_col, brief_col = st.columns([0.35, 0.65])
    with score_col:
        st.plotly_chart(plot_score(score), use_container_width=True)
        # st.success(f"Niveau : {crisis_level(score)}")
        # On récupère le texte du niveau
        niveau_texte = crisis_level(score)
        
        # Changement de couleur dynamique selon le score (ou le mot-clé)
        if score < 40 or "faible" in niveau_texte.lower():
            st.success(f"Niveau : {niveau_texte}")  # Boîte verte
        elif score < 70 or "moyen" in niveau_texte.lower() or "modéré" in niveau_texte.lower():
            st.warning(f"Niveau : {niveau_texte}")  # Boîte orange/jaune
        else:
            st.error(f"Niveau : {niveau_texte}")    # Boîte rouge
    with brief_col:
        st.markdown(result["brief_deterministe"])

    if use_openrouter:
        api_key = get_api_key()
        try:
            with st.spinner("Génération du brief OpenRouter..."):
                llm_brief = generate_openrouter_brief(result, api_key=api_key, model=openrouter_model)
            st.markdown("### Brief rédigé par LLM")
            st.markdown(llm_brief)
        except Exception as e:
            st.warning(f"Brief LLM indisponible. Brief déterministe conservé. Détail : {e}")

    st.markdown("---")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Narratifs sur la période")
        st.dataframe(result["narratives"], use_container_width=True)
    with right:
        st.markdown("#### Top hashtags")
        st.dataframe(result["top_hashtags"], use_container_width=True)

    st.markdown("#### Top auteurs")
    st.dataframe(result["top_authors"], use_container_width=True)

    st.markdown("#### Top posts")
    st.dataframe(result["top_posts"], use_container_width=True)

    st.plotly_chart(plot_timeline_hourly(result["hourly_timeline"]), use_container_width=True)

with tab4:
    st.subheader("Agent 2 — Cartographie des narratifs et acteurs")
    result = ensure_active_result(agent, peaks_d, top_n)
    st.info("Agent 2 utilise le diagnostic Agent 1 actif. Il affine les narratifs, les acteurs par narratif, les mutations et les angles de réponse.")

    if st.button("Lancer / mettre à jour Agent 2", key="run_agent2"):
        st.session_state["agent2_report"] = AgentNarratifsActeurs().analyze(result, time_bucket="hour", top_n=top_n)
    agent2_report = st.session_state.get("agent2_report") or ensure_agent2(result, top_n)

    st.markdown(agent2_report["markdown"])
    left, right = st.columns(2)
    with left:
        st.markdown("#### Matrice de risque narratif")
        st.dataframe(agent2_report["risk_matrix"], use_container_width=True)
    with right:
        st.markdown("#### Acteurs par narratif")
        st.dataframe(agent2_report["actor_narratives"], use_container_width=True)

    st.markdown("#### Mutations narratives détectées")
    st.dataframe(agent2_report["mutation_signals"], use_container_width=True)

    tl = agent2_report["narrative_timeline"]
    if len(tl):
        time_col = "hour" if "hour" in tl.columns else "day"
        top_narr = agent2_report["risk_matrix"]["narrative"].head(5).tolist() if len(agent2_report["risk_matrix"]) else []
        plot_tl = tl[tl["narrative"].isin(top_narr)] if top_narr else tl
        fig = px.line(plot_tl, x=time_col, y="messages", color="narrative", markers=True, title="Évolution des narratifs dominants")
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Agent 3 — Propagation et coordination prudente")
    result = ensure_active_result(agent, peaks_d, top_n)
    st.info("Agent 3 mesure la vitesse, la concentration des relais, les copier-coller et les pics horaires. Il parle de signaux, jamais de preuve de bots.")

    if st.button("Lancer / mettre à jour Agent 3", key="run_agent3"):
        st.session_state["agent3_report"] = AgentPropagationCoordination().analyze(result, top_n=top_n)
    agent3_report = st.session_state.get("agent3_report") or ensure_agent3(result, top_n)

    coord = agent3_report["coordination_score"]
    cscore, ctext = st.columns([0.35, 0.65])
    with cscore:
        st.plotly_chart(plot_score(coord["score"], title="Coordination prudente"), use_container_width=True)
    with ctext:
        st.markdown(agent3_report["markdown"])

    left, right = st.columns(2)
    with left:
        st.markdown("#### Top comptes de concentration")
        conc = agent3_report["concentration_metrics"]
        st.dataframe(conc.get("top_authors_table", pd.DataFrame()), use_container_width=True)
    with right:
        st.markdown("#### Copier-coller / gabarits")
        st.dataframe(agent3_report["copy_patterns"], use_container_width=True)

    st.markdown("#### Pics horaires")
    st.dataframe(agent3_report["burst_patterns"], use_container_width=True)

with tab6:
    st.subheader("Agent 4 — Stratège de riposte")
    result = ensure_active_result(agent, peaks_d, top_n)
    agent2_report = ensure_agent2(result, top_n)
    agent3_report = ensure_agent3(result, top_n)
    ctx = build_crisis_context(result)
    st.info(f"Agent 4 utilise Agent 1 + Agent 2 + Agent 3 : {ctx['periode'].get('start')} → {ctx['periode'].get('end')} | {ctx['niveau']} | narratif dominant : {ctx['dominant_narrative']}")

    col1, col2 = st.columns(2)
    with col1:
        objective = st.selectbox(
            "Objectif principal",
            [
                "Rétablir la confiance et clarifier les faits",
                "Réduire l'emballement viral sans sur-réagir",
                "Expliquer les règles d'attribution et la gouvernance",
                "Protéger les personnes ciblées et condamner les menaces",
                "Préparer une réponse média courte et maîtrisée",
            ],
        )
        posture = st.selectbox(
            "Posture",
            [
                "Automatique selon la crise",
                "Transparence + pédagogie",
                "Factuel + apaisement",
                "Empathique + sécurité",
                "Neutralité institutionnelle",
            ],
        )
    with col2:
        cible = st.multiselect(
            "Cibles",
            ["Grand public", "Médias", "Créateurs concernés", "Tutelle / ministère", "Équipes internes", "Partenaires"],
            default=["Grand public", "Médias", "Équipes internes"],
        )
        risk_tolerance = st.selectbox("Tolérance au risque", ["Très prudente", "Prudente", "Réactive", "Offensive mais factuelle"], index=1)
        response_window = st.selectbox("Fenêtre d'action", ["Prochaines 6 heures", "Prochaines 24 heures", "Prochains 3 jours"], index=1)

    strategist = AgentStrategieRiposte()
    strategy = build_strategy_safe(
        strategist,
        result,
        objective=objective,
        posture=posture,
        cible=cible,
        risk_tolerance=risk_tolerance,
        response_window=response_window,
        agent2_report=agent2_report,
        agent3_report=agent3_report,
    )
    st.session_state["agent4_strategy"] = strategy

    st.markdown(strategy["markdown"])

    if use_openrouter:
        api_key = get_api_key()
        try:
            with st.spinner("Réécriture stratégique OpenRouter..."):
                llm_strategy = generate_openrouter_strategy(result, strategy, api_key=api_key, model=openrouter_model)
            st.markdown("### Version LLM — stratégie améliorée")
            st.markdown(llm_strategy)
            st.session_state["agent4_strategy_llm"] = llm_strategy
        except Exception as e:
            st.warning(f"Stratégie LLM indisponible. Détail : {e}")

with tab7:
    st.subheader("Agent 5 — Rédacteur + garde-fou")
    result = ensure_active_result(agent, peaks_d, top_n)
    agent2_report = ensure_agent2(result, top_n)
    agent3_report = ensure_agent3(result, top_n)
    if "agent4_strategy" not in st.session_state:
        st.session_state["agent4_strategy"] = build_strategy_safe(
            AgentStrategieRiposte(),
            result,
            agent2_report=agent2_report,
            agent3_report=agent3_report,
        )
    strategy = st.session_state["agent4_strategy"]

    col1, col2 = st.columns(2)
    with col1:
        tone = st.selectbox("Ton", ["Institutionnel, neutre et apaisant", "Très court et factuel", "Pédagogique et transparent", "Empathique et protecteur"])
        spokesperson = st.text_input("Porte-parole / entité", value="Le CNC")
    with col2:
        channels = st.multiselect(
            "Formats à générer",
            ["Post X court", "Thread X", "Communiqué", "FAQ", "Réponse journaliste", "Message interne", "Message d'attente"],
            default=["Post X court", "Communiqué", "FAQ", "Réponse journaliste", "Message interne"],
        )

    redacteur = AgentRedacteurGardeFou()
    draft_pack = redacteur.generate_pack(result, strategy, tone=tone, channels=channels, spokesperson=spokesperson)
    st.session_state["agent5_drafts"] = draft_pack

    st.markdown(draft_pack["markdown"])
    st.markdown("### Tableau garde-fou")
    st.dataframe(draft_pack["guardrail"], use_container_width=True)

    if use_openrouter:
        api_key = get_api_key()
        try:
            with st.spinner("Réécriture des messages OpenRouter..."):
                llm_drafts = generate_openrouter_drafts(result, strategy, draft_pack, api_key=api_key, model=openrouter_model)
            st.markdown("### Version LLM — messages améliorés")
            st.markdown(llm_drafts)
            st.session_state["agent5_drafts_llm"] = llm_drafts
        except Exception as e:
            st.warning(f"Messages LLM indisponibles. Détail : {e}")

with tab8:
    st.subheader("Orchestration Top 1 — pipeline complet")
    result = ensure_active_result(agent, peaks_d, top_n)
    agent2_report = ensure_agent2(result, top_n)
    agent3_report = ensure_agent3(result, top_n)
    strategy = build_strategy_safe(
        AgentStrategieRiposte(),
        result,
        agent2_report=agent2_report,
        agent3_report=agent3_report,
    )
    drafts = AgentRedacteurGardeFou().generate_pack(result, strategy)
    st.session_state["agent4_strategy"] = strategy
    st.session_state["agent5_drafts"] = drafts

    st.markdown(
        """
        ```text
        Corpus X/Twitter
              ↓
        Agent 1 — Diagnostic : chiffres, pics, acteurs, score de crise
              ↓
        Agent 2 — Narratifs : matrice de risque, mutations, acteurs par narratif
              ↓
        Agent 3 — Propagation : vitesse, concentration, copier-coller, coordination prudente
              ↓
        Agent 4 — Stratégie : posture, plan 0–24h, risques, déclencheurs
              ↓
        Agent 5 — Rédaction : messages + garde-fou + validation humaine
              ↓
        Agent 6 — Chatbot : répond aux questions à partir des sorties Agents 1 à 5
        ```
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 1️⃣ Diagnostic")
        st.markdown(result["brief_deterministe"])
    with c2:
        st.markdown("### 2️⃣ Narratifs + Propagation")
        st.markdown(agent2_report["markdown"])
        st.markdown(agent3_report["markdown"])
    with c3:
        st.markdown("### 3️⃣ Riposte")
        st.markdown(strategy["markdown"])
        st.markdown(drafts["markdown"])

    st.success("Démo prête : l'IA observe, cartographie, mesure, recommande, rédige, répond aux questions, puis l'humain valide.")

with tab9:
    st.subheader("Agent 6 — Chatbot connecté aux résultats")
    st.info(
        "Pose n'importe quelle question sur la crise, les chiffres, les narratifs, la propagation, "
        "la stratégie ou les messages. L'Agent 6 répond en priorité avec les sorties des Agents 1 à 5."
    )

    # 1. Récupération des données (Ceci s'exécute au premier chargement)
    result = ensure_active_result(agent, peaks_d, top_n)
    agent2_report = ensure_agent2(result, top_n)
    agent3_report = ensure_agent3(result, top_n)

    if "agent4_strategy" not in st.session_state:
        st.session_state["agent4_strategy"] = build_strategy_safe(
            AgentStrategieRiposte(),
            result,
            agent2_report=agent2_report,
            agent3_report=agent3_report,
        )
    strategy = st.session_state["agent4_strategy"]

    if "agent5_drafts" not in st.session_state:
        st.session_state["agent5_drafts"] = AgentRedacteurGardeFou().generate_pack(result, strategy)
    draft_pack = st.session_state["agent5_drafts"]

    # Construction du contexte envoyé au LLM
    agent6_context = build_agent6_context(
        result,
        agent2_report=agent2_report,
        agent3_report=agent3_report,
        strategy=strategy,
        draft_pack=draft_pack,
    )

    # Affichage des métriques de l'onglet
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Base prioritaire", "Agents 1 à 5")
    with c2:
        st.metric("Mode", "LLM OpenRouter" if use_openrouter else "Déterministe")
    with c3:
        st.metric("Validation", "Humaine obligatoire")

    # 2. Appel du fragment interactif
    interface_chatbot_agent6(agent6_context, use_openrouter, openrouter_model)
    
with tab10:
    st.subheader("Exports prêts pour slides / GitHub / Jour 3")
    result = ensure_active_result(agent, peaks_d, top_n)
    agent2_report = st.session_state.get("agent2_report")
    agent3_report = st.session_state.get("agent3_report")
    strategy = st.session_state.get("agent4_strategy")
    draft_pack = st.session_state.get("agent5_drafts")

    files = {}
    files.update(result_to_export_files(result, df, prefix="agent1_streamlit"))
    files.update(agent23_export_files(agent2_report, agent3_report))
    files.update(agent45_export_files(strategy, draft_pack))
    if st.session_state.get("agent6_history"):
        conv_md = "\n\n".join(
            [
                f"## Question {i}\n{t['question']}\n\n### Réponse\n{t['answer']}"
                for i, t in enumerate(st.session_state["agent6_history"], start=1)
            ]
        )
        files["agent6_conversation.md"] = conv_md.encode("utf-8")

    zip_bytes = make_agent1_zip(files)
    st.download_button(
        "⬇️ Télécharger tous les exports Agents 1-6 (.zip)",
        data=zip_bytes,
        file_name="exports_crisisai_agents_1_6.zip",
        mime="application/zip",
    )
    st.markdown("Fichiers inclus :")
    st.write(sorted(files.keys()))

    st.markdown("#### Télécharger le corpus enrichi seul")
    enriched_cols = [
        "dt", "author", "engagement_type", "sentiment", "likes", "comments", "shares", "reach",
        "impressions", "is_retweet", "is_reply", "is_quote", "is_original", "main_narrative",
        "narratives", "risk_level", "text_raw", "text_norm",
    ]
    export_df = df[[c for c in enriched_cols if c in df.columns]].copy()
    if "narratives" in export_df.columns:
        export_df["narratives"] = export_df["narratives"].astype(str)
    st.download_button(
        "⬇️ Télécharger corpus_enrichi_agents.csv",
        data=export_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
        file_name="corpus_enrichi_agents.csv",
        mime="text/csv",
    )

