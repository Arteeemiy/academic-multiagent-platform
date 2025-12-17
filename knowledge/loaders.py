from pathlib import Path
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PDF_DIR = PROJECT_ROOT / "knowledge" / "sources" / "pdf"


def load_pdf_knowledge():
    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        raise RuntimeError(f"No PDF files found in {PDF_DIR}")

    relative_paths = [str(Path("sources") / "pdf" / p.name) for p in pdf_files]

    return PDFKnowledgeSource(file_paths=relative_paths)


def load_knowledge_sources():
    """
    Single entry point for Crew knowledge.
    Easy to extend later (PDF + JSON + String).
    """
    sources = []
    sources.append(load_pdf_knowledge())
    return sources
