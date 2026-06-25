# Script oral — Agents 1, 4 et 5

## Introduction

Nous avons simplifié l'architecture pour garder les agents les plus utiles au Datathon :

- **Agent 1** : observe et diagnostique la crise.
- **Agent 4** : transforme le diagnostic en stratégie de riposte.
- **Agent 5** : rédige les messages et active un garde-fou avant publication.

## Démo

1. J'importe le corpus.
2. L'Agent 1 détecte les pics, les narratifs, les acteurs moteurs et le score de crise.
3. Je sélectionne le pic principal.
4. L'Agent 4 prend ce brief et propose une posture de réponse, un plan 0–24 h et les risques à éviter.
5. L'Agent 5 génère un post X, un communiqué, une FAQ et une réponse journaliste.
6. Le garde-fou vérifie les risques : attaque personnelle, accusation non prouvée, promesse non validée.
7. On termine par l'orchestration complète Agent 1 → Agent 4 → Agent 5.

## Pourquoi c'est agentique ?

Ce n'est pas juste une page avec des graphiques :

- l'Agent 1 lit le corpus et calcule ;
- l'Agent 4 prend une décision de stratégie à partir du diagnostic ;
- l'Agent 5 produit une action concrète : des messages ;
- le système garde une validation humaine avant publication.

## Phrase forte

> Notre IA ne publie pas. Elle analyse, priorise, rédige et alerte. La décision finale reste humaine, ce qui est essentiel en communication de crise.

## Limites honnêtes

- Les narratifs sont détectés par mots-clés dans cette version Jour 2 : c'est rapide, interprétable et robuste.
- On ne dit pas qu'il y a des bots : on parle de dynamique d'amplification et de signaux faibles.
- Les messages doivent être validés par une cellule humaine.
