# forest-rag

Système de question-réponse basé sur RAG (Retrieval-Augmented Generation),
appliqué à un corpus documentaire sur la conception de jardins-forêts et
l'agroforesterie tempérée.

## Objectif

Projet personnel construit pour pratiquer un pipeline RAG complet, du
prototype au déploiement : indexation vectorielle, appel direct à une API
LLM (sans framework d'abstraction), exposition via API REST, et suivi
basique de performance (latence, coût par requête).

## Stack technique

- **Indexation** : chunking + embeddings (text-embedding-3-small / bge-m3)
- **Stockage vectoriel** : ChromaDB
- **Génération** : appel direct API LLM (OpenAI / Mistral)
- **Serving** : FastAPI
- **Déploiement** : Render
- **Monitoring** : logging requêtes, latence, tokens/coût

## Corpus

Corpus de documents libres en français sur la permaculture et les
jardins-forêts (sources et licences détaillées dans `/data/SOURCES.md`).

## Limitations connues

- Corpus de littérature pratique/associative, non académique (pas de
  validation peer-review)
- Projet de démonstration, pas un système en production avec utilisateurs réels
- Pas de réévaluation continue ni de fine-tuning du retrieval
