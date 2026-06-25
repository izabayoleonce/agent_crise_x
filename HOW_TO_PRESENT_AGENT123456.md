# Script oral — CrisisAI War Room Agents 1 à 5

## Introduction

Nous avons transformé l'analyse Jour 1 en une mini War Room IA. L'objectif n'est pas de publier automatiquement, mais d'aider une cellule de communication à comprendre la crise, prioriser les risques et préparer une riposte validable humainement.

## Démo

1. Nous importons le corpus X/Twitter.
2. L'Agent 1 diagnostique la crise : volume, pics, acteurs, narratifs, score de crise.
3. L'Agent 2 cartographie les narratifs : il identifie les thèmes dominants, les acteurs qui les portent et les mutations narratives.
4. L'Agent 3 analyse la propagation : vitesse, concentration, copier-coller et signaux prudents d'amplification synchronisée.
5. L'Agent 4 transforme ces résultats en stratégie de riposte.
6. L'Agent 5 rédige des messages et applique un garde-fou avant validation humaine.

## Pourquoi c'est agentique ?

Chaque agent a un rôle spécialisé :

- Agent 1 observe et calcule.
- Agent 2 interprète les narratifs.
- Agent 3 mesure la propagation.
- Agent 4 décide de la posture.
- Agent 5 produit les messages et contrôle les risques.

## Phrase forte

Ce n'est pas un simple dashboard. C'est une chaîne agentique de gestion de crise : comprendre, prioriser, décider, rédiger et valider.

## Limites assumées

- Les narratifs par mots-clés sont interprétables mais peuvent être affinés par embeddings/LLM.
- Le score de coordination ne prouve pas des bots : il signale des patterns à vérifier.
- L'IA assiste la communication, mais ne publie jamais seule.


# Ajout Agent 6 — Chatbot War Room

L'Agent 6 sert à montrer que notre solution n'est pas seulement une suite de tableaux. C'est un assistant utilisable en cellule de crise : on peut lui poser une question libre, et il répond en priorité à partir des résultats des Agents 1 à 5.

Phrase à dire :
> L'Agent 6 est notre interface conversationnelle. Il ne remplace pas les agents précédents : il les rend exploitables. Il répond aux incompréhensions du décideur en s'appuyant sur le diagnostic, les narratifs, la propagation, la stratégie et les messages déjà produits.

Démo rapide :
1. Générer l'analyse Agent 1 sur le pic principal.
2. Ouvrir Agent 6.
3. Poser : « Que doit faire le CNC dans les prochaines 24h ? ».
4. Montrer que la réponse cite la stratégie, les risques et les garde-fous.
