# arxiv-paper-summary

Génération automatique de **sommaires de papiers de recherche arXiv** (catégorie cs.AI), via un pipeline NLP en Python.

Le terme « sommaire » étant ambigu, le projet répond aux **deux interprétations** :

- **Sommaire de contenu** (résumé synthétique) — `summarize_arxiv.py`
- **Sommaire structurel** (table des matières) — `extract_toc_arxiv.py`

Ce dépôt est un POC réalisé sur une demi-journée de travail dans le cadre d'un cas d'usage technique. Il privilégie la **clarté du raisonnement** sur l'exhaustivité du code. Le document `methodologie.md` détaille la démarche, le choix des approches, l'évaluation et les limites assumées.

---

## Aperçu de l'approche

```
[arXiv API]  →  [Téléchargement PDF]  →  [Parsing PyMuPDF]  →  [Pipeline NLP]  →  [Sommaire]
```

### Sommaire de contenu — `summarize_arxiv.py`

Pipeline hybride **section-aware** :

1. Détection des sections (Introduction, Method, Results, Conclusion).
2. Résumé section par section avec **PEGASUS-arxiv** (modèle abstractif local pré-entraîné spécifiquement sur arXiv).
3. Composition du sommaire final structuré.
4. **Fallback TextRank** (extractif, non-supervisé) si la détection de sections échoue.

### Sommaire structurel — `extract_toc_arxiv.py`

Cascade de trois stratégies par ordre de confiance :

1. **Bookmarks PDF natifs** (`fitz.get_toc()`).
2. **Détection par numérotation** (regex sur les en-têtes du type `1. Introduction`, `2.1 Related Work`).
3. **Détection par taille de police** (heuristique typographique, fallback).

---

## Installation

```bash
git clone https://github.com/<votre-pseudo>/arxiv-paper-summary.git
cd arxiv-paper-summary
pip install -r requirements.txt
```

Python 3.10+ recommandé.

---

## Usage

### Génération d'un sommaire de contenu

```bash
python summarize_arxiv.py --query "cat:cs.AI" --max 3 --out ./summaries
```

Options :

| Flag | Défaut | Description |
|---|---|---|
| `--query` | `cat:cs.AI` | requête arXiv (syntaxe API arXiv) |
| `--max` | `3` | nombre de papiers à traiter |
| `--out` | `./summaries` | dossier de sortie |

### Extraction d'une table des matières

```bash
python extract_toc_arxiv.py --query "cat:cs.AI" --max 3 --out ./toc
```

Premier lancement : le téléchargement du modèle PEGASUS prend 1-2 minutes (~2 Go).

---

## Exemples de sortie

Le dossier `examples/` contient quelques sommaires réellement générés sur des papiers arXiv récents.

---

## Stack technique

- **Python 3.10+**
- [`arxiv`](https://pypi.org/project/arxiv/) — API arXiv officielle
- [`PyMuPDF`](https://pymupdf.readthedocs.io/) (`fitz`) — parsing PDF avec accès aux métadonnées de police
- [`transformers`](https://huggingface.co/docs/transformers/index) — modèle PEGASUS-arxiv (HuggingFace)
- [`sumy`](https://github.com/miso-belica/sumy) — TextRank pour le fallback extractif

---

## Méthodologie complète

Voir [`methodologie.md`](./methodologie.md) pour le document détaillé :

- Analyse du problème et reformulation
- Comparaison de quatre familles d'approches
- Justification du choix retenu
- Plan d'évaluation (ROUGE, BERTScore, faithfulness via NLI, évaluation humaine)
- Limites assumées et perspectives

---

## Limites assumées

- Détection de sections par heuristique fragile sur les papiers atypiques (mitigée par un fallback TextRank).
- PEGASUS limité à 1024 tokens en entrée → map-reduce pour les sections longues.
- Formules mathématiques rendues en pseudo-LaTeX peuvent polluer le résumé.
- Pas de gestion des figures (un papier dont le résultat-clé est un graphique sera mal résumé).
- Pas d'évaluation quantitative implémentée dans la fenêtre de temps.

---

## Pistes d'amélioration

- Remplacer la détection de sections par GROBID (parseur scientifique XML TEI).
- Comparer empiriquement BART, PEGASUS et un LLM commercial (GPT-4o, Claude) sur 50 papiers.
- Implémenter l'évaluation ROUGE / BERTScore / faithfulness annoncée dans la méthodologie.
- Ajouter une dimension multilingue (FR/EN).
- Containeriser et industrialiser pour un volume cible.

---

## Auteur

Nikola Kostadinovic — Data Scientist & Chef de projet IA
