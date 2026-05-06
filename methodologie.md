# Cas d'usage AFNOR — Génération de sommaires de papiers de recherche arXiv

**Auteur :** Nikola Kostadinovic
**Date :** mai 2026
**Contexte :** Cas d'usage proposé pour le second entretien Data Scientist & Chef de projet IA — AFNOR

---

## 1. Analyse du problème

### 1.1 Reformulation

Le problème consiste à concevoir une **méthode automatisée** capable, à partir d'une collection de PDFs de papiers de recherche issus d'arXiv (catégorie cs.AI), de produire pour chacun un **sommaire** lisible synthétisant l'essentiel du contenu.

### 1.2 Clarifier ce qu'on entend par « sommaire »

Le terme est volontairement ambigu. Avant d'entrer en solution, je distingue deux interprétations possibles :

- **Sommaire structurel (table des matières)** : la liste des sections du document (Abstract, Introduction, Method, Results, Conclusion). Tâche d'extraction structurée.
- **Sommaire de contenu (résumé / synthèse)** : une vue condensée de la substance du papier — problème abordé, méthode, résultats, contributions. Tâche de résumé automatique.

**J'ai traité les deux** plutôt que d'arbitrer arbitrairement, parce que les deux sont utiles dans la vie réelle et qu'ils mobilisent des techniques très différentes :

- `extract_toc_arxiv.py` → sommaire structurel (parsing PDF, détection d'en-têtes, hiérarchie)
- `summarize_arxiv.py` → sommaire de contenu (NLP, résumé abstractif section-aware)

Le reste de cette méthodologie détaille principalement le **sommaire de contenu**, qui est la tâche la plus riche techniquement et la plus alignée avec les enjeux SMART Standards d'AFNOR. La section 8 décrit brièvement l'approche structurelle.

### 1.3 Définition du succès

Un bon sommaire automatique répond à plusieurs critères :

- **Fidélité (faithfulness)** : il ne dit rien que le papier ne dise. Pas d'hallucination.
- **Couverture** : il aborde les points essentiels (problème, méthode, résultats principaux).
- **Concision** : 5 à 10 phrases typiquement, lisible en 30 secondes.
- **Structure** : compréhensible sans lire le papier.
- **Robustesse** : fonctionne sur la diversité des papiers cs.AI (théorie, applicatif, survey, etc.).

### 1.4 Spécificités du corpus arXiv cs.AI

Les papiers arXiv ont des propriétés exploitables :

- **Structure relativement standardisée** : Abstract, Introduction, Related Work, Method, Experiments, Results, Conclusion, References.
- **Présence systématique d'un Abstract** rédigé par les auteurs — c'est *déjà* un résumé humain de référence.
- **Format LaTeX → PDF**, donc parsing souvent propre (titres, sections clairement marqués).
- **Longueur variable** : 6 à 30+ pages, avec des sections de tailles très inégales.
- **Vocabulaire technique dense**, formules mathématiques, tableaux, figures.

Cette régularité structurelle est **un levier important** : on peut s'appuyer dessus plutôt que traiter les papiers comme du texte brut.

---

## 2. Approches envisagées et comparaison

J'identifie quatre familles d'approches, classées par maturité technique. L'objectif de cette section est de comparer leurs trade-offs, pas de choisir d'avance.

### 2.1 Approche A — Extractive « non-supervisée » (TextRank, LSA)

**Principe :** sélectionner les phrases du document qui sont les plus représentatives, sans rien générer. TextRank construit un graphe de similarité entre phrases et applique PageRank.

**Avantages**
- Pas de modèle à entraîner, pas de GPU, pas d'API.
- 100 % fidèle au texte source (par construction, on n'invente rien).
- Implémentation rapide (`sumy`, `gensim`).

**Inconvénients**
- Le résumé est une concaténation de phrases hors contexte → souvent incohérent à la lecture.
- Aucune compréhension sémantique : pas de capacité à reformuler ou structurer.
- Sur un papier scientifique, sélectionne souvent des phrases triviales (« In this paper we propose… »).

### 2.2 Approche B — Abstractive avec modèle pré-entraîné (BART, T5, PEGASUS)

**Principe :** un modèle séquence-à-séquence pré-entraîné sur des tâches de résumé (CNN/DailyMail, XSum) génère un texte nouveau.

**Avantages**
- Texte cohérent, reformulé, lisible.
- Pas de coût d'API, exécutable en local sur GPU modeste, voire CPU.
- Modèles disponibles directement sur Hugging Face (`facebook/bart-large-cnn`, `google/pegasus-arxiv` qui est *spécifiquement* pré-entraîné sur arXiv).

**Inconvénients**
- **Limite de contexte** : BART ~1024 tokens, PEGASUS-arxiv ~1024 tokens, T5 ~512 tokens. Or un papier arXiv fait 5 000 à 30 000 tokens. Il faut donc une stratégie pour gérer la longueur.
- Risque d'hallucination, surtout si on truncate brutalement.
- Le style du résumé reflète celui du dataset d'entraînement (souvent journalistique).

### 2.3 Approche C — Abstractive via LLM avec map-reduce ou refine

**Principe :** utiliser un LLM moderne (GPT-4o, Claude, Mistral) avec une stratégie de découpage pour gérer la longueur. Deux variantes :

- **Map-reduce** : découper le papier en chunks, résumer chaque chunk, puis résumer les résumés.
- **Refine** : résumer chunk 1, puis pour chaque chunk suivant raffiner le résumé courant à la lumière du nouveau chunk.

**Avantages**
- Qualité de génération supérieure (cohérence, raisonnement).
- Contexte étendu (128k tokens chez GPT-4o ou Claude → un papier entier tient en une seule fenêtre la plupart du temps).
- Contrôle fin via prompt engineering (style, longueur, structure imposée).

**Inconvénients**
- Coût d'API (faible mais non nul à l'échelle de centaines de papiers).
- Dépendance à un fournisseur externe → enjeu de souveraineté pour AFNOR.
- Sensibilité au prompt : un prompt mal calibré peut générer un sommaire générique.

### 2.4 Approche D — Hybride section-aware

**Principe :** exploiter la structure du papier. Détecter les sections (Abstract, Method, Results, Conclusion), traiter chacune avec un prompt dédié, puis composer le sommaire final.

**Avantages**
- Tire parti de la régularité du corpus arXiv.
- Sommaire structuré dès la conception (1 phrase problème, 2 phrases méthode, 2 phrases résultats…).
- Plus interprétable : on sait d'où vient chaque morceau du résumé.
- Réduit les hallucinations en localisant la source de chaque affirmation.

**Inconvénients**
- Demande un parser PDF qui détecte les sections (heuristiques fragiles, dépend du LaTeX d'origine).
- Sensible aux papiers atypiques (surveys, théoriques) qui ne suivent pas la structure standard.

### 2.5 Tableau de synthèse

| Critère | A — Extractive | B — BART/PEGASUS | C — LLM map-reduce | D — Hybride section-aware |
|---|---|---|---|---|
| Fidélité | ★★★★★ | ★★★ | ★★★★ | ★★★★ |
| Lisibilité | ★★ | ★★★★ | ★★★★★ | ★★★★★ |
| Coût compute | ★ | ★★ | ★★★ (API) | ★★★ |
| Implémentation rapide | ★★★★★ | ★★★★ | ★★★ | ★★ |
| Robustesse aux papiers atypiques | ★★★★★ | ★★★★ | ★★★★ | ★★ |
| Contrôle de structure | ★ | ★★ | ★★★★ | ★★★★★ |
| Souveraineté (local possible) | ★★★★★ | ★★★★★ | ★★ | ★★★ (selon LLM) |

---

## 3. Choix retenu et justification

Compte tenu **du temps imparti (une demi-journée), de la contrainte « pas d'API LLM » que je me suis fixée pour rester reproductible, et de la volonté d'un livrable qui montre une vraie démarche**, je propose une architecture **hybride** :

> **Approche D (section-aware) avec un modèle abstractif local (Approche B, PEGASUS-arxiv) appliqué section par section, et un fallback sur Approche A (TextRank) si la détection de sections échoue.**

### Pourquoi ce choix

1. **Tirer parti de la structure** : un papier arXiv n'est pas du texte brut, c'est un document hiérarchisé. Ne pas l'exploiter serait gaspiller un signal fort.
2. **PEGASUS pré-entraîné sur arXiv** : Google a publié un modèle spécifiquement entraîné sur ce corpus. C'est l'outil le plus adapté disponible localement, sans coût d'API.
3. **Robustesse** : le fallback TextRank garantit qu'on produit *quelque chose* même sur les papiers atypiques où la détection de sections échoue.
4. **Évaluabilité** : avec une structure section-by-section, on peut comparer chaque morceau du résumé à la section source — ce qui facilite la mesure de fidélité.
5. **Cohérence avec AFNOR** : c'est exactement le pattern qui s'appliquerait au RAG normatif — exploiter la structure des normes (sections, articles), pas les traiter comme du texte plat.

### Ce que cela donne en sortie

```
Titre : <titre du papier>
Auteurs : <liste>
Date : <date arXiv>

Problème abordé : <1 phrase générée depuis Introduction>
Méthode proposée : <2 phrases générées depuis Method>
Principaux résultats : <2 phrases générées depuis Results>
Conclusion / implications : <1 phrase générée depuis Conclusion>
```

---

## 4. Architecture de la solution

### 4.1 Schéma général

```
[arXiv API]
    ↓ recherche cs.AI récents
[Téléchargement PDF]
    ↓ liste de PDFs
[Parser PDF (PyMuPDF)]
    ↓ texte avec coordonnées et tailles de police
[Détection de sections]
    ↓ dictionnaire {section_name: text}
[Pour chaque section pertinente]
    ↓ chunking si > limite tokens
[PEGASUS-arxiv summarization]
    ↓ phrases résumées par section
[Composition du sommaire final]
    ↓
[Sortie structurée + métadonnées]
```

### 4.2 Détection de sections

Approche heuristique simple, suffisante pour la plupart des papiers arXiv :

- Détecter les en-têtes par taille de police + casse (TITRE EN MAJUSCULES, ou texte plus gros que le corps).
- Mapper les titres détectés sur un référentiel : `Abstract`, `Introduction`, `Method` (variantes : *Methodology*, *Approach*, *Model*), `Experiments` / `Results`, `Conclusion`.
- Si moins de 3 sections détectées → fallback TextRank sur l'ensemble.

PyMuPDF (`fitz`) expose les métadonnées de police, ce qui rend cette détection viable.

### 4.3 Stratégie de gestion de la longueur

PEGASUS-arxiv a une limite de 1024 tokens en entrée. Stratégies par section :

- Section courte (< 1024 tokens) → un seul appel.
- Section longue → chunking en blocs de 800 tokens avec 100 tokens de chevauchement, summarization de chaque chunk, puis concaténation et reformulation finale (mini map-reduce).

### 4.4 Stack technique retenue

- **Python 3.11**
- **arxiv** (PyPI) pour l'API arXiv.
- **PyMuPDF** (`fitz`) pour le parsing PDF avec accès aux métadonnées de police.
- **Hugging Face transformers** (`google/pegasus-arxiv` ou `facebook/bart-large-cnn` en alternative).
- **sumy** pour le fallback TextRank.
- **tqdm** pour le suivi.

Pas de Docker ici (gain de temps), mais une `requirements.txt` claire pour la reproductibilité.

---

## 5. Évaluation — comment je mesurerais la qualité

L'évaluation d'un système de résumé est notoirement difficile. Sur ce volume et dans ce délai, je propose une approche en trois temps.

### 5.1 Métriques automatiques

- **ROUGE-1, ROUGE-2, ROUGE-L** : comparer le sommaire généré à l'**Abstract** du papier comme référence (l'Abstract étant un résumé écrit par les auteurs eux-mêmes). Métrique classique mais imparfaite.
- **BERTScore** : similarité sémantique plutôt que lexicale, plus pertinent pour des sommaires reformulés.

### 5.2 Métriques sémantiques modernes

- **Faithfulness** : pour chaque phrase du sommaire, vérifier qu'on retrouve son contenu dans la section source. Implémentable via un modèle NLI (Natural Language Inference) sur paires (phrase_résumé, phrase_source).

### 5.3 Évaluation humaine sur un échantillon

Sur 10-20 papiers, faire évaluer manuellement (ou demander à un LLM en mode juge) sur 4 critères :

1. Couverture (les points essentiels sont-ils présents ?)
2. Fidélité (rien d'inventé ?)
3. Cohérence (lisible sans lire le papier ?)
4. Concision (longueur appropriée ?)

C'est l'évaluation qui compte le plus dans la vie réelle.

---

## 6. Regard critique sur cette approche

Il est aussi important de pointer mes propres limites que de défendre mes choix.

### 6.1 Limites assumées

- **La détection de sections par heuristique de police est fragile** : un papier atypique (workshop, format custom) cassera l'approche. C'est pour ça que j'ai prévu le fallback TextRank.
- **PEGASUS est un modèle de 2020** : performant sur arXiv mais probablement moins fluide qu'un GPT-4o moderne en abstractive.
- **Pas d'évaluation quantitative dans cette demi-journée** : j'ai posé le cadre méthodologique mais je n'ai pas implémenté ROUGE / BERTScore — ce serait l'étape suivante.
- **Pas de gestion fine des formules mathématiques** : PyMuPDF les rend en pseudo-LaTeX qui peut polluer le résumé. Stratégie possible : remplacer par `[FORMULA]` à l'extraction.
- **Pas de gestion des figures** : un papier où le résultat-clé est une figure est mal résumé par cette approche. Améliorations possibles via vision (GPT-4V, Claude Vision).

### 6.2 Ce que je ferais avec plus de temps

1. Comparer empiriquement BART, PEGASUS, et un LLM (GPT-4o-mini en API) sur 50 papiers, mesurer ROUGE et faithfulness.
2. Implémenter une vraie évaluation humaine sur un échantillon, avec une grille à 4 critères.
3. Tester un parser PDF plus avancé (GROBID, qui est spécialisé sur les papiers scientifiques et reconstruit la structure XML TEI).
4. Ajouter une dimension multilingue (papiers FR / EN — utile pour AFNOR avec les normes).
5. Construire un benchmark interne (papiers + résumés humains de référence) pour itérer.

### 6.3 Ce que cette approche démontre vis-à-vis du poste AFNOR

- **Réflexe d'exploiter la structure documentaire** — exactement la posture pour les normes AFNOR.
- **Conscience des trade-offs** entre approches simples et sophistiquées, locales et cloud, fidèles et fluides.
- **Souci de la mesurabilité** : on ne livre pas un système, on livre un système *évaluable*.
- **Pragmatisme** : un fallback TextRank pour les cas atypiques, plutôt qu'une solution parfaite qui casse 10 % du temps.

---

## 8. Sommaire structurel — l'autre lecture du brief

L'approche complémentaire `extract_toc_arxiv.py` répond à l'interprétation « table des matières ». La tâche est moins ambiguë, plus déterministe, mais elle a aussi ses pièges.

### 8.1 Trois stratégies en cascade

Plutôt qu'une seule méthode, j'ai choisi une **cascade de fallbacks** classée par confiance :

1. **Bookmarks PDF natifs** (`fitz.get_toc()`) — la voie royale quand le PDF en contient. Rapide, fiable, gratuit. Mais peu de papiers arXiv ont des bookmarks intégrés (LaTeX → pdflatex sans `hyperref` correctement configuré).
2. **Détection par numérotation** — regex sur les en-têtes de la forme `1. Introduction`, `2.1 Related Work`, `III. Method`. C'est l'usage massivement dominant sur arXiv : les papiers sont structurés selon les conventions ICML / NeurIPS / ACL. Le niveau hiérarchique se déduit du nombre de points dans la numérotation.
3. **Détection par typographie** — heuristique sur la taille de police : on identifie la taille modale (corps), puis on classe les lignes plus grandes en niveaux. Fragile (peut capter des légendes de figures) mais utile en dernier recours.

### 8.2 Pourquoi une cascade plutôt qu'une seule méthode

Mêmes raisons que pour le pipeline section-aware : **un système robuste vaut mieux qu'un système élégant qui casse 10 % du temps**. Sur 100 papiers, on s'attend à avoir une couverture proche de 100 % :
- ~10 % de bookmarks natifs (stratégie 1)
- ~80 % de numérotation détectable (stratégie 2)
- ~10 % de typographie (stratégie 3)

Le sommaire généré indique explicitement la stratégie utilisée, ce qui permet de tracer la confiance.

### 8.3 Limites du sommaire structurel

- Pas de distinction entre sections « techniques » (Method, Results) et sections « de service » (References, Appendix). En production on filtrerait par liste blanche.
- Pas de gestion multilingue — la regex est anglocentrée.
- GROBID ferait un travail bien plus précis (parsing scientifique XML TEI) mais demande de déployer un service Java à part. Trade-off à arbitrer selon le volume cible.

### 8.4 Quand utiliser l'un ou l'autre ?

- **Sommaire structurel** : navigation, indexation, génération de hyperliens, vue d'ensemble rapide. Utilité forte pour SMART Standards (un utilisateur veut savoir si une norme couvre son sujet avant de la lire).
- **Sommaire de contenu** : aide à la décision (vaut-il la peine de lire ?), digest de veille, onboarding. Utilité forte pour les rédacteurs de normes en phase d'analyse.

**Les deux sont complémentaires**, pas concurrents — c'est aussi un point de discussion intéressant en entretien.

---

## 9. Annexe — Pour aller plus loin

### Idées d'extensions naturelles

- **Sommaires multi-papiers** : à partir de plusieurs papiers sur un même thème, générer une synthèse de l'état de l'art (cas d'usage typique pour les rédacteurs de normes AFNOR qui font de la veille).
- **Sommaires personnalisés** : adapter la longueur et la profondeur au profil utilisateur (chercheur vs ingénieur vs décideur).
- **Détection des contributions principales** : extraire spécifiquement les *claims* (« We show that… », « Our method outperforms… ») via des patterns ou un classifieur fine-tuné.
- **Lien avec un RAG** : indexer les sommaires plutôt que les papiers entiers pour accélérer la recherche, et utiliser le papier complet seulement quand on cite.

### Références

- Cohan et al., *A Discourse-Aware Attention Model for Abstractive Summarization of Long Documents*, NAACL 2018 — sur la summarization de papiers arXiv.
- Zhang et al., *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization*, ICML 2020.
- Lewis et al., *BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation*, ACL 2020.
- Sumy library — TextRank et autres méthodes extractives.

---

*Document rédigé en demi-journée. Le code squelette associé se trouve dans `summarize_arxiv.py`.*
