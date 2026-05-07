"""
Génération de sommaires de papiers arXiv (cs.AI).

Pipeline section-aware avec PEGASUS-arxiv et fallback TextRank pour les
papiers où la détection de sections échoue. Voir le document de méthodologie
associé pour la justification des choix.

Installation : pip install arxiv pymupdf transformers torch sumy nltk
Usage        : python summarize_arxiv.py --query "large language models" --max 5
"""

from __future__ import annotations
import argparse
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import arxiv
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)


SECTION_KEYWORDS = {
    "abstract":     ["abstract"],
    "introduction": ["introduction", "1. introduction"],
    "method":       ["method", "methodology", "approach", "model", "our approach", "proposed"],
    "results":      ["results", "experiments", "evaluation", "experimental results"],
    "conclusion":   ["conclusion", "conclusions", "discussion", "discussion and conclusion"],
}

SUMMARY_SECTIONS = ["introduction", "method", "results", "conclusion"]


@dataclass
class PaperMeta:
    arxiv_id: str
    title: str
    authors: list[str]
    published: str
    pdf_url: str
    pdf_path: Path | None = None


def fetch_arxiv_papers(query: str = "cs.AI", max_results: int = 5,
                       output_dir: Path = Path("./papers")) -> list[PaperMeta]:
    """Récupère les papiers les plus récents matchant la query via l'API arXiv."""
    output_dir.mkdir(exist_ok=True, parents=True)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    for result in search.results():
        meta = PaperMeta(
            arxiv_id=result.entry_id.split("/")[-1],
            title=result.title.strip(),
            authors=[a.name for a in result.authors],
            published=result.published.isoformat(),
            pdf_url=result.pdf_url,
        )
        meta.pdf_path = output_dir / f"{meta.arxiv_id}.pdf"
        if not meta.pdf_path.exists():
            log.info(f"[fetch] {meta.arxiv_id} - {meta.title[:60]}")
            result.download_pdf(dirpath=str(output_dir), filename=meta.pdf_path.name)
        papers.append(meta)
    return papers


@dataclass
class ParsedPaper:
    meta: PaperMeta
    full_text: str
    sections: dict[str, str] = field(default_factory=dict)


def parse_pdf(meta: PaperMeta) -> ParsedPaper:
    """Extrait le texte du PDF et tente de détecter les sections.

    On utilise PyMuPDF (fitz) pour ses métadonnées de police, qui permettent
    de repérer les en-têtes par leur taille (plus grosse que le corps).
    """
    doc = fitz.open(meta.pdf_path)
    full_text_parts = []
    candidate_headers = []  # (page_idx, font_size, text)

    for page_idx, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if block.get("type") != 0:  # 0 = texte
                continue
            for line in block["lines"]:
                line_text = " ".join(span["text"] for span in line["spans"]).strip()
                if not line_text:
                    continue
                full_text_parts.append(line_text)
                max_font_size = max(span["size"] for span in line["spans"])
                # Ligne courte + grosse police = en-tête probable
                if len(line_text) < 80 and max_font_size > 11:
                    candidate_headers.append((page_idx, max_font_size, line_text))

    full_text = "\n".join(full_text_parts)
    sections = _split_by_headers(full_text, candidate_headers)
    return ParsedPaper(meta=meta, full_text=full_text, sections=sections)


def _split_by_headers(full_text: str, headers: list[tuple[int, float, str]]) -> dict[str, str]:
    """Découpe le texte en sections nommées via le référentiel SECTION_KEYWORDS."""
    sections: dict[str, str] = {}
    text_lower = full_text.lower()

    matches = []  # (start_pos, section_name)
    for sec_name, keywords in SECTION_KEYWORDS.items():
        for kw in keywords:
            pattern = rf"\n\s*\d?\.?\s*{re.escape(kw)}\s*\n"
            m = re.search(pattern, text_lower)
            if m:
                matches.append((m.start(), sec_name))
                break  # première occurrence par section

    matches.sort(key=lambda x: x[0])
    for i, (start, name) in enumerate(matches):
        end = matches[i + 1][0] if i + 1 < len(matches) else len(full_text)
        sections[name] = full_text[start:end].strip()

    return sections


class Summarizer:
    """Wrapper PEGASUS-arxiv. Modèle chargé une seule fois (coût initial lourd)."""

    def __init__(self, model_name: str = "google/pegasus-arxiv"):
        log.info(f"[summarizer] chargement de {model_name}")
        from transformers import PegasusTokenizer, PegasusForConditionalGeneration
        self.tokenizer = PegasusTokenizer.from_pretrained(model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(model_name)
        self.max_input_tokens = 1024

    def summarize_text(self, text: str, max_target_tokens: int = 100) -> str:
        """Résume un texte qui tient en une fenêtre du modèle."""
        inputs = self.tokenizer(
            text,
            max_length=self.max_input_tokens,
            truncation=True,
            return_tensors="pt",
        )
        outputs = self.model.generate(
            **inputs,
            max_length=max_target_tokens,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def summarize_long(self, text: str, max_target_tokens: int = 120) -> str:
        """Map-reduce pour les textes au-delà de la fenêtre de 1024 tokens."""
        token_count = len(self.tokenizer.encode(text))
        if token_count <= self.max_input_tokens:
            return self.summarize_text(text, max_target_tokens)

        # Chunking par phrases pour éviter de couper en plein milieu
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks, current, current_len = [], [], 0
        for s in sentences:
            s_len = len(self.tokenizer.encode(s, add_special_tokens=False))
            if current_len + s_len > 800 and current:  # marge sous 1024
                chunks.append(" ".join(current))
                current, current_len = [s], s_len
            else:
                current.append(s)
                current_len += s_len
        if current:
            chunks.append(" ".join(current))

        chunk_summaries = [self.summarize_text(c, 80) for c in chunks]
        return self.summarize_text(" ".join(chunk_summaries), max_target_tokens)


def textrank_fallback(text: str, num_sentences: int = 6) -> str:
    """Résumé extractif par TextRank — utilisé quand la détection de sections échoue."""
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, num_sentences)
    return " ".join(str(s) for s in sentences)


def compose_summary(parsed: ParsedPaper, summarizer: Summarizer) -> str:
    """Compose le sommaire final. Fallback TextRank si moins de 2 sections détectées."""
    detected = [s for s in SUMMARY_SECTIONS if s in parsed.sections]
    log.info(f"[compose] sections détectées : {detected}")

    if len(detected) < 2:
        log.warning("[compose] détection insuffisante → fallback TextRank")
        body = textrank_fallback(parsed.full_text, num_sentences=6)
        return _format_output(parsed.meta, body, structured=False)

    section_summaries = {}
    for sec in detected:
        text = parsed.sections[sec]
        if len(text.split()) < 30:  # section trop courte, on saute
            continue
        log.info(f"[compose] résumé section {sec}")
        section_summaries[sec] = summarizer.summarize_long(text, max_target_tokens=100)

    return _format_output(parsed.meta, section_summaries, structured=True)


def _format_output(meta: PaperMeta, body, structured: bool) -> str:
    lines = [
        f"# {meta.title}",
        f"**Auteurs :** {', '.join(meta.authors[:3])}{' et al.' if len(meta.authors) > 3 else ''}",
        f"**Date :** {meta.published[:10]}",
        f"**arXiv :** {meta.arxiv_id}",
        "",
        "## Sommaire",
    ]
    if structured:
        labels = {
            "introduction": "Problème abordé",
            "method":       "Méthode proposée",
            "results":      "Principaux résultats",
            "conclusion":   "Conclusion / implications",
        }
        for sec, summary in body.items():
            lines.append(f"\n### {labels.get(sec, sec.title())}")
            lines.append(summary)
    else:
        lines.append("\n*(résumé extractif TextRank — détection de sections insuffisante)*")
        lines.append(body)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Résumé automatique de papiers arXiv cs.AI")
    parser.add_argument("--query", default="cat:cs.AI", help="requête arXiv")
    parser.add_argument("--max", type=int, default=3, help="nombre de papiers à traiter")
    parser.add_argument("--out", default="./summaries", help="dossier de sortie")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(exist_ok=True, parents=True)

    log.info("=== Récupération arXiv ===")
    papers = fetch_arxiv_papers(query=args.query, max_results=args.max)

    log.info("=== Chargement du modèle ===")
    summarizer = Summarizer()

    for paper in papers:
        log.info(f"=== {paper.title[:80]} ===")
        try:
            parsed = parse_pdf(paper)
            summary = compose_summary(parsed, summarizer)
            out_file = output_dir / f"{paper.arxiv_id}_summary.md"
            out_file.write_text(summary, encoding="utf-8")
            log.info(f"[ok] sommaire écrit : {out_file}")
        except Exception as e:
            log.error(f"[fail] {paper.arxiv_id} : {e}")
            continue


if __name__ == "__main__":
    main()
