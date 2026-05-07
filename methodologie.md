# Cas d'usage AFNOR — Sommaires de papiers arXiv

**Nikola Kostadinovic — mai 2026**

## Le problème

À partir d'une collection de PDFs de papiers arXiv (cs.AI), produire pour chacun un sommaire lisible — au sens d'un résumé synthétique du contenu (problème abordé, méthode, résultats), pas d'une simple table des matières.

Un bon résumé automatique doit être fidèle (rien d'inventé), couvrir l'essentiel (problème, méthode, résultats), tenir en quelques phrases, et rester lisible sans avoir lu le papier. Sur arXiv on a un avantage : les papiers ont une structure assez standardisée (Abstract, Introduction, Method, Results, Conclusion) et un Abstract rédigé par les auteurs, donc une référence humaine "gratuite".

## Approches considérées

J'ai regardé quatre familles avant de trancher.

**Extractive type TextRank.** On sélectionne les phrases les plus représentatives sans rien générer. Avantage massif : aucune hallucination par construction, et c'est trivial à implémenter avec `sumy`. Inconvénient : sur un papier scientifique, ça sort souvent des phrases banales ("In this paper we propose...") et le résultat se lit mal.

**Abstractive avec un modèle pré-entraîné (BART, PEGASUS).** Un seq2seq génère un texte nouveau. PEGASUS a une variante spécifiquement entraînée sur arXiv, ce qui est intéressant. Le souci, c'est la limite de contexte (~1024 tokens), alors qu'un papier en fait 5 000 à 30 000. Il faut donc une stratégie pour gérer la longueur.

**LLM moderne avec map-reduce ou refine.** Qualité supérieure, contexte étendu (128k tokens chez GPT-4o ou Claude — un papier tient en une fenêtre), contrôle fin par prompt. Le coût d'API n'est pas un blocage à petite échelle, mais la dépendance à un fournisseur externe est un vrai sujet pour AFNOR (souveraineté).

**Hybride section-aware.** Exploiter la structure du papier : détecter les sections, traiter chacune avec un prompt dédié, composer le sommaire final. Plus interprétable, plus facile à évaluer (on sait d'où vient chaque phrase du résumé), mais ça suppose un parser qui détecte correctement les sections, ce qui est fragile sur les papiers atypiques.

## Le choix

Je suis parti sur du **section-aware avec PEGASUS-arxiv en local**, et un fallback TextRank quand la détection de sections échoue.

Plusieurs raisons. D'abord, ne pas exploiter la structure d'un papier arXiv serait du gâchis. Un papier n'est pas du texte plat. La même logique vaut pour les normes AFNOR — elles aussi sont structurées en articles, sections, paragraphes — donc le pattern est transposable.

Ensuite, PEGASUS-arxiv existe et tourne en local. Pas d'API, pas de coût marginal, reproductible. C'est l'outil le plus adapté disponible pour ce corpus précis.

Le fallback TextRank n'est pas accessoire : il garantit qu'on produit quelque chose même quand le parser de sections plante. Mieux vaut un système un peu moins beau qui marche tout le temps qu'un système élégant qui casse une fois sur dix.

Enfin, le découpage section par section facilite l'évaluation. On peut comparer chaque morceau du résumé à la section dont il vient, ce qui rend la fidélité mesurable.

La sortie visée par papier :

```
Titre, auteurs, date.
Problème abordé : 1 phrase (depuis l'Introduction).
Méthode proposée : 2 phrases (depuis Method).
Résultats : 2 phrases (depuis Results/Experiments).
Conclusion : 1 phrase.
```

## Architecture

Le pipeline est linéaire :

1. Récupération des PDFs via l'API arXiv.
2. Parsing avec PyMuPDF (`fitz`). Cette lib donne accès aux métadonnées de police, ce qui sert pour la détection de sections.
3. Détection des sections par heuristique : on repère les en-têtes par taille/casse, on les map sur un référentiel (Abstract, Introduction, Method/Methodology/Approach, Experiments/Results, Conclusion). Si moins de trois sections sont détectées, fallback TextRank sur le document entier.
4. Pour chaque section pertinente, summarization avec PEGASUS. Si la section dépasse 1024 tokens, on chunk avec recouvrement (800 tokens, 100 de chevauchement) et on fait un mini map-reduce.
5. Composition du sommaire final selon le gabarit ci-dessus.

Stack : Python 3.11, `arxiv` pour l'API, `PyMuPDF` pour le PDF, `transformers` (Hugging Face) pour PEGASUS, `sumy` pour le fallback. Pas de Docker dans le délai imparti mais un `requirements.txt` propre.

## Évaluation

C'est le sujet le plus inconfortable en summarization, parce qu'il n'y a pas de vérité unique.

À court terme, je m'appuierais sur ROUGE-1/2/L en comparant le sommaire généré à l'Abstract du papier (qui sert de référence humaine, métrique imparfaite mais standard), BERTScore pour capturer la similarité sémantique au-delà du lexical, et un score de faithfulness via NLI : pour chaque phrase du sommaire, vérifier qu'on retrouve son contenu dans la section source via un modèle d'inférence.

À moyen terme, une vraie **évaluation humaine** sur 10-20 papiers (couverture, fidélité, cohérence, concision). C'est la seule qui compte vraiment, mais elle ne tient pas dans une demi-journée.

## Limites assumées

La détection de sections par taille de police est fragile. Un papier atypique cassera l'approche, d'où le fallback.

PEGASUS date de 2020. Un GPT-4o moderne ferait probablement mieux en fluidité, c'est un compromis assumé pour rester local.

Je n'ai pas implémenté l'évaluation quantitative dans le délai. Le cadre est posé, l'exécution viendrait ensuite.

Les formules mathématiques sont mal gérées (PyMuPDF rend du pseudo-LaTeX qui pollue). Une stratégie simple serait de les remplacer par `[FORMULA]` à l'extraction. Les figures ne sont pas exploitées du tout — pour un papier où le résultat-clé est une figure, le résumé sera pauvre. Une vraie amélioration passerait par un modèle vision.

## Ce que je ferais avec plus de temps

Comparer empiriquement BART, PEGASUS et un LLM (GPT-4o-mini) sur 50 papiers, mesurer ROUGE et faithfulness pour avoir des chiffres. Tester GROBID, qui est un parser scientifique spécialisé et reconstruit la structure XML TEI — bien plus solide que mes heuristiques. Ajouter une dimension multilingue FR/EN, utile pour les normes. Construire un petit benchmark interne avec des résumés humains de référence pour itérer.

## Pour finir

Ce qui me semble important au-delà du livrable : exploiter la structure documentaire plutôt que traiter le PDF comme du texte plat, assumer les compromis (local vs cloud, fidèle vs fluide, simple vs sophistiqué), et livrer un système évaluable plutôt qu'un système opaque. Le fallback TextRank est moins glamour qu'un pipeline parfait, mais c'est ce qui distingue une preuve de concept robuste d'une démo fragile.

---

*Document écrit en demi-journée. Code dans `summarize_arxiv.py`.*
