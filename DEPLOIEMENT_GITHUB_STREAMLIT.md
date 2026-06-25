# Déploiement GitHub + Streamlit Cloud

## 1. Vérifier qu'aucune clé API n'est dans le dossier

Sur Windows PowerShell :

```powershell
Get-ChildItem -Recurse -File | Select-String -Pattern "sk-or-v1"
```

Si une clé apparaît, supprime-la du fichier concerné avant de pousser sur GitHub.

## 2. Envoyer le projet sur GitHub

```bash
git init
git add .
git commit -m "CrisisAI War Room agents 1 a 5"
git branch -M main
git remote add origin https://github.com/TON_USER/TON_REPO.git
git push -u origin main --force
```

Si `origin already exists` :

```bash
git remote set-url origin https://github.com/TON_USER/TON_REPO.git
git push -u origin main --force
```

## 3. Déployer sur Streamlit Cloud

- Va sur Streamlit Cloud.
- Clique sur **New app**.
- Sélectionne ton repo GitHub.
- `Main file path` : `app.py`.
- Clique sur **Deploy**.

## 4. Ajouter la clé OpenRouter, optionnel

Dans Streamlit Cloud → App → Settings → Secrets :

```toml
OPENROUTER_API_KEY = "your_openrouter_api_key_here"
```

Ne mets jamais la vraie clé dans `.streamlit/secrets.toml.example`.

## 5. Tester l'app

- Importe `data.xlsx`.
- Va dans **Agent 1 — Diagnostic**.
- Analyse le pic principal.
- Ouvre **Agent 2**, **Agent 3**, **Agent 4**, **Agent 5**.
- Ouvre **Orchestration Top 1** pour la démo finale.
