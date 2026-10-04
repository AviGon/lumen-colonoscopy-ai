"""Polyp review MCP server for the report-writing agents.

segment_polyp runs the trained U-Net in ``models/`` via polyp_inference.py; search_pubmed gives
the agents literature to cite. Outputs must not be used for diagnosis or clinical decision making.
"""

from __future__ import annotations

import os
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from fastmcp import FastMCP

mcp = FastMCP("Polyp Review Tools")

PROJECT_ROOT = Path(os.environ.get("POLYP_PROJECT_ROOT", Path(__file__).parent)).resolve()
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PUBMED_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PUBMED_TOOL_NAME = "polyp_review_mcp"
PUBMED_EMAIL = os.environ.get("NCBI_EMAIL", "local@example.invalid")


def _project_path(path_value: str) -> Path:
    """Resolve an image or data path and reject traversal outside this project."""
    candidate = Path(path_value)
    path = (PROJECT_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    try:
        path.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError("Path must be inside the project directory.") from exc
    return path


def _image_path(path_value: str) -> Path:
    path = _project_path(path_value)
    if not path.exists() or not path.is_file():
        raise ValueError(f"Image file does not exist: {path_value}")
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image type: {path.suffix or 'no extension'}")
    return path


def _display_path(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def load_model_if_available(model_name: str | None = None) -> dict[str, Any]:
    """Locate a PyTorch checkpoint in ``models/``; polyp_inference.py loads it."""
    if not MODELS_DIR.exists():
        return {"available": False, "reason": "models directory does not exist", "artifact": None}

    artifacts = sorted(path for path in MODELS_DIR.iterdir() if path.is_file() and path.suffix in {".pth", ".pt"})
    if model_name:
        artifacts = [path for path in artifacts if path.stem == model_name or path.name == model_name]
    artifact = artifacts[0] if artifacts else None
    return {
        "available": artifact is not None,
        "artifact": str(artifact) if artifact else None,
        "reason": "checkpoint found" if artifact else "no model checkpoint found",
    }


@mcp.tool()
def segment_polyp(image_path: str, threshold: float = 0.5) -> dict[str, Any]:
    """Segment one image with the trained U-Net and write report artifacts.

    Writes input.png, mask.png, overlay.png, prob_heatmap.png and features.json to
    reports/<image name>/ and returns the extracted features. Research use only.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")
    path = _image_path(image_path)
    loader = load_model_if_available()
    if not loader["available"]:
        raise ValueError(f"Segmentation model unavailable: {loader['reason']}")

    from polyp_inference import run_inference  # deferred: torch import is slow

    out_dir = REPORTS_DIR / path.stem
    features = run_inference(str(path), str(out_dir), threshold=threshold, ckpt=loader["artifact"])
    return {
        "model_status": "trained",
        "clinical_notice": (
            "Research model output (test set mean Dice 0.906). Not clinically validated; "
            "must not be used for diagnosis or treatment decisions."
        ),
        "image_path": _display_path(path),
        "output_dir": _display_path(out_dir),
        "artifacts": {
            name: _display_path(out_dir / name)
            for name in ("input.png", "mask.png", "overlay.png", "prob_heatmap.png", "features.json")
        },
        "features": features,
    }


def _pubmed_request(endpoint: str, params: dict[str, str]) -> bytes:
    query = urllib.parse.urlencode({**params, "tool": PUBMED_TOOL_NAME, "email": PUBMED_EMAIL})
    request = urllib.request.Request(
        f"{PUBMED_BASE_URL}/{endpoint}?{query}",
        headers={"User-Agent": f"{PUBMED_TOOL_NAME}/0.1 ({PUBMED_EMAIL})"},
    )
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310, NCBI HTTPS endpoint
        return response.read()


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _parse_articles(xml_bytes: bytes) -> list[dict[str, str]]:
    root = ET.fromstring(xml_bytes)
    articles: list[dict[str, str]] = []
    for citation in root.findall(".//PubmedArticle"):
        article = citation.find(".//Article")
        if article is None:
            continue
        abstract_parts = [_node_text(part) for part in article.findall(".//Abstract/AbstractText")]
        abstract = "\n".join(part for part in abstract_parts if part)
        journal = _node_text(article.find(".//Journal/Title"))
        pub_date = article.find(".//Journal/JournalIssue/PubDate")
        date = " ".join(
            value
            for value in (
                _node_text(pub_date.find("Year")) if pub_date is not None else "",
                _node_text(pub_date.find("Month")) if pub_date is not None else "",
                _node_text(pub_date.find("Day")) if pub_date is not None else "",
            )
            if value
        )
        articles.append(
            {
                "pmid": _node_text(citation.find(".//PMID")),
                "title": _node_text(article.find("ArticleTitle")),
                "journal": journal,
                "publication_date": date,
                "abstract": abstract,
            }
        )
    return articles


@mcp.tool()
def search_pubmed(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search PubMed and return bibliographic records with plain-text abstracts only."""
    query = query.strip()
    if not query:
        raise ValueError("query cannot be empty.")
    if not 1 <= max_results <= 20:
        raise ValueError("max_results must be between 1 and 20.")

    try:
        search_xml = _pubmed_request(
            "esearch.fcgi", {"db": "pubmed", "term": query, "retmax": str(max_results), "retmode": "xml"}
        )
        ids = [node.text for node in ET.fromstring(search_xml).findall(".//Id") if node.text]
        if not ids:
            return {"query": query, "count": 0, "articles": [], "message": "No PubMed records found."}
        fetch_xml = _pubmed_request(
            "efetch.fcgi", {"db": "pubmed", "id": ",".join(ids), "retmode": "xml"}
        )
        articles = _parse_articles(fetch_xml)
        return {
            "query": query,
            "count": len(articles),
            "articles": articles,
            "content_type": "text_abstracts_only",
            "message": "Results include bibliographic metadata and text abstracts only. No PDFs are retrieved.",
        }
    except (urllib.error.URLError, TimeoutError, ET.ParseError) as exc:
        return {"query": query, "count": 0, "articles": [], "error": f"PubMed request failed: {exc}"}


if __name__ == "__main__":
    mcp.run()
