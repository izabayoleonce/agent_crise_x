# CrisisAI War Room — Agents IA 1 à 5

Application Streamlit complète pour le Datathon **“Gérer une crise virale avec des agents IA”**.

L'app transforme le notebook Agent 1 en une mini **War Room IA** déployable : elle importe un corpus X/Twitter (`data.xlsx`, `.csv`, `.xls`), analyse la crise, cartographie les narratifs, mesure la propagation, propose une stratégie de riposte et rédige des messages avec garde-fous.

## Architecture complète

```text
Corpus X/Twitter
      ↓
Agent 1 — Analyste de crise
      ↓
Agent 2 — Cartographe des narratifs et acteurs
      ↓
Agent 3 — Propagation et coordination prudente
      ↓
Agent 4 — Stratège de riposte
      ↓
Agent 5 — Rédacteur + garde-fou
      ↓
Messages prêts à validation humaine
```

## Rôle des agents

| Agent | Rôle | Sorties principales |
|---|---|---|
| Agent 1 | Diagnostic global | KPIs, pics, top auteurs, top hashtags, Crisis Velocity Score, brief |
| Agent 2 | Narratifs + acteurs | Matrice de risque narratif, acteurs par narratif, mutations narratives |
| Agent 3 | Propagation | Vitesse, concentration, copier-coller, pics horaires, coordination prudente |
| Agent 4 | Stratégie | Posture, plan 0–24h, risques, messages prioritaires |
| Agent 5 | Rédaction + garde-fou | Post X, communiqué, FAQ, réponse journaliste, message interne, contrôle risques |

## Structure du repo

```text
.
├── app.py
├── streamlit_app.py
├── requirements.txt
├── src/
│   ├── agent1_core.py
│   ├── agent23_core.py
│   ├── agent45_core.py
│   └── llm_clients.py
├── prompts/
│   ├── agent1_openrouter_prompt.md
│   ├── agent2_narratifs_prompt.md
│   ├── agent3_propagation_prompt.md
│   ├── agent4_strategie_prompt.md
│   └── agent5_redaction_gardefou_prompt.md
├── notebooks/
│   └── agent-1-datathon.ipynb
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── DEPLOIEMENT_GITHUB_STREAMLIT.md
└── HOW_TO_PRESENT_AGENT12345.md
```

## Lancer en local

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py
```

## Déployer sur Streamlit Cloud

1. Créer un repo GitHub.
2. Mettre tous les fichiers du dossier dans le repo.
3. Aller sur Streamlit Cloud.
4. Choisir le repo.
5. Main file path : `app.py`.
6. Déployer.

## OpenRouter optionnel

L'application fonctionne sans OpenRouter : les calculs et les rapports déterministes sont produits par Python/Pandas.

OpenRouter sert uniquement à reformuler certains briefs en style plus professionnel.

Dans Streamlit Cloud, ajoute dans **Settings → Secrets** :

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
```

Ne mets jamais ta vraie clé API dans GitHub.

## Démo recommandée Jour 2 / Jour 3

1. Importer `data.xlsx`.
2. Montrer les KPIs globaux.
3. Ouvrir **Agent 1** et sélectionner le pic principal.
4. Ouvrir **Agent 2** : montrer le narratif prioritaire et la matrice de risque.
5. Ouvrir **Agent 3** : montrer le score de coordination prudente et rappeler que ce n'est pas une accusation de bots.
6. Ouvrir **Agent 4** : montrer la stratégie de riposte.
7. Ouvrir **Agent 5** : montrer les messages et le garde-fou.
8. Terminer par **Orchestration Top 1**.

## Phrase Top 1

> Ce n'est pas un simple dashboard. C'est une mini War Room IA : elle observe la crise, cartographie les narratifs, mesure la propagation, propose une stratégie et rédige des messages validables par une cellule humaine.


## Version corrigée finale

Cette version est synchronisée pour le pipeline complet **Agent 1 → Agent 2 → Agent 3 → Agent 4 → Agent 5**.

Points corrigés :

- `AgentStrategieRiposte.build_strategy()` accepte `agent2_report` et `agent3_report`.
- `app.py` contient une fonction de compatibilité `build_strategy_safe()` pour éviter le crash si un ancien cache Streamlit charge une ancienne version du fichier.
- L’Agent 4 est donc capable d’utiliser les sorties des Agents 2 et 3 pour renforcer la stratégie de riposte.
- Aucune vraie clé OpenRouter n’est incluse dans le repo.

Si Streamlit garde une ancienne erreur en cache, arrête l’app, supprime `__pycache__`, puis relance :

```powershell
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
streamlit run app.py
```



## Agent 6 — Chatbot War Room

La version finale ajoute un **Agent 6 conversationnel**. Il permet de poser des questions libres comme :

- Pourquoi le pic principal est-il important ?
- Quels narratifs sont les plus risqués ?
- Y a-t-il des signaux de coordination ?
- Quelle stratégie faut-il adopter ?
- Rédige un post X prudent.
- Quelles sont les limites de l'analyse ?

L'Agent 6 répond en priorité à partir des sorties des Agents 1 à 5. OpenRouter est optionnel : sans clé API, une réponse déterministe est produite ; avec la clé, le LLM reformule la réponse de manière plus professionnelle à partir du contexte JSON vérifié.
