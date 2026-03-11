
"""Research Utilities and Tools.

This module provides search and content processing utilities for the research agent,
including web search capabilities and content summarization tools.
"""

import re
import os
import json
import hashlib
import math
import time
import threading
import urllib.error
import urllib.request
from urllib.parse import urlparse
import html as html_lib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from typing_extensions import Annotated, Literal
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg, tool
from tavily import TavilyClient
from FlagEmbedding import FlagReranker

from src.model_config import (
    PIPELINE_MODEL_SETTINGS,
    build_chat_model,
)
from src.state_research import Summary
from src.prompts import summarize_webpage_prompt, report_generation_with_draft_insight_prompt, article_summary_prompt
from src.templates import format_template_for_prompt
from src.logging_config import get_logger, log_timing, log_token_usage, get_global_tracker
from src.pdf_processor import (
    PdfProcessorConfig,
    DependencyError as PdfDependencyError,
    chunk_text as chunk_pdf_text,
    extract_pdf_to_markdown,
)

# ===== LOGGING SETUP =====

logger = get_logger("deep_research.utils")
_SOURCE_METADATA_START = "<!-- SOURCE_METADATA_START -->"
_SOURCE_METADATA_END = "<!-- SOURCE_METADATA_END -->"
_PLACEHOLDER_TITLE_RE = re.compile(r"^[#>*\-\s]*t?itle[:\s]*$", re.IGNORECASE)

# ===== UTILITY FUNCTIONS =====

def get_today_str() -> str:
    """Get current date in a human-readable format."""
    return datetime.now().strftime("%a %b %-d, %Y")

def get_current_dir() -> Path:
    """Get the current directory of the module.

    This function is compatible with Jupyter notebooks and regular Python scripts.

    Returns:
        Path object representing the current directory
    """
    try:
        return Path(__file__).resolve().parent
    except NameError:  # __file__ is not defined
        return Path.cwd()

# ===== CONFIGURATION =====

summarization_model = build_chat_model(PIPELINE_MODEL_SETTINGS.webpage_summarization_model)
writer_model = build_chat_model(PIPELINE_MODEL_SETTINGS.draft_refinement_model)
article_summary_model = build_chat_model(PIPELINE_MODEL_SETTINGS.article_ingest_summary_model)
MAX_CONTEXT_LENGTH = 250000 # max context length (characters) for the summarization model
DEFAULT_DOC_ID = "user_article"

# Token-based PDF chunking configuration (docling VLM + table-aware splitter)
PDF_PROCESSOR_CONFIG = PdfProcessorConfig()

# Persistent cache (saves OCR + chunking + summary artifacts across runs)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / ".deep_research_cache"
CACHE_ENABLED = os.environ.get("DEEP_RESEARCH_CACHE", "true").lower() != "false"
INGEST_CACHE_VERSION = "v1"  # bump when OCR prompt/chunking/summary logic changes

# In-memory chunk store for reranking: doc_id -> List[Document]
DOCUMENT_CHUNKS: Dict[str, List[Document]] = {}
# Metadata store for global summaries: doc_id -> {"global_summary": str}
DOCUMENT_METADATA: Dict[str, Dict[str, Any]] = {}
_STORE_LOCK = threading.Lock()
_RERANKER_LOCK = threading.Lock()
_RERANKER_INIT_LOCK = threading.Lock()

_RERANKER: Optional["FlagReranker"] = None
_RERANKER_MODEL_NAME = PIPELINE_MODEL_SETTINGS.document_chunk_reranker_model.model_name
_TAVILY_CLIENT: Optional[TavilyClient] = None

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_text(text: str) -> str:
    return _sha256_bytes(text.encode("utf-8", errors="ignore"))

def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)

def _write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)

def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_reranker_device() -> str:
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Torch is required for the FlagEmbedding reranker but is not installed."
        ) from exc

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _reset_reranker() -> None:
    global _RERANKER
    with _RERANKER_INIT_LOCK:
        _RERANKER = None


def _get_tavily_client() -> TavilyClient:
    """Lazily initialize the Tavily client when web search is actually used."""
    global _TAVILY_CLIENT
    if _TAVILY_CLIENT is None:
        _TAVILY_CLIENT = TavilyClient()
    return _TAVILY_CLIENT


def _get_reranker(
    model_name: str = _RERANKER_MODEL_NAME,
    force_device: Optional[str] = None,
) -> "FlagReranker":
    """Lazily initialize and return a global FlagReranker instance."""
    global _RERANKER
    if _RERANKER is not None:
        return _RERANKER

    with _RERANKER_INIT_LOCK:
        if _RERANKER is not None:
            return _RERANKER

        device = force_device or _select_reranker_device()
        use_fp16 = device == "cuda"
        logger.info(
            "Initializing reranker model %s on device %s (use_fp16=%s)",
            model_name,
            device,
            use_fp16,
        )
        reranker = FlagReranker(model_name, use_fp16=use_fp16, device=device)
        _RERANKER = reranker
        return reranker


def _is_meta_tensor_device_error(exc: Exception) -> bool:
    normalized = str(exc or "").lower()
    return "meta tensor" in normalized and ("to_empty" in normalized or "no data" in normalized)


def _compute_reranker_scores(reranker: "FlagReranker", pairs: List[List[str]]) -> Any:
    with _RERANKER_LOCK:
        try:
            return reranker.compute_score(pairs, batch_size=16, normalize=True)
        except AttributeError:
            return reranker.predict(pairs, batch_size=16)  # type: ignore[call-arg]


def _normalize_reranker_scores(scores: Any) -> Tuple[List[float], int]:
    if isinstance(scores, float):
        raw_scores = [scores]
    elif isinstance(scores, list):
        raw_scores = list(scores)
    else:
        raw_scores = list(scores)  # type: ignore[arg-type]

    normalized_scores: List[float] = []
    non_finite_count = 0
    for raw in raw_scores:
        try:
            value = float(raw)
        except Exception:
            value = float("nan")
        if not math.isfinite(value):
            non_finite_count += 1
            value = 0.0
        normalized_scores.append(value)
    return normalized_scores, non_finite_count

# ===== SEARCH FUNCTIONS =====

def tavily_search_multiple(
    search_queries: List[str], 
    max_results: int = 3, 
    topic: Literal["general", "news", "finance"] = "general", 
    include_raw_content: bool = True, 
) -> List[dict]:
    """Perform search using Tavily API for multiple queries.

    Args:
        search_queries: List of search queries to execute
        max_results: Maximum number of results per query
        topic: Topic filter for search results
        include_raw_content: Whether to include raw webpage content

    Returns:
        List of search result dictionaries
    """
    logger.debug("TAVILY_SEARCH_MULTIPLE | queries=%d, max_results=%d, topic=%s", 
                len(search_queries), max_results, topic)

    # Execute searches sequentially. Note: you can use AsyncTavilyClient to parallelize this step.
    search_docs = []
    total_results = 0
    
    for i, query in enumerate(search_queries):
        query_preview = query[:100] + "..." if len(query) > 100 else query
        logger.debug("  query[%d]: %s", i, query_preview)
        
        start_time = time.perf_counter()
        try:
            #result = tavily_client.search(
            #    query,
            #    max_results=max_results,
            #    include_raw_content=include_raw_content,
            #    topic=topic
            #)
            result = _get_tavily_client().search(
                query=query,
                search_depth="advanced",
                max_results=max_results,
                include_raw_content=include_raw_content,
                include_answer="advanced",  # or True, or "basic"
                topic=topic,
                chunks_per_source=3
            )
            elapsed = time.perf_counter() - start_time
            
            num_results = len(result.get('results', []))
            total_results += num_results
            logger.debug("    -> %d results in %.2fs", num_results, elapsed)
            
            search_docs.append(result)
            
        except Exception as e:
            elapsed = time.perf_counter() - start_time
            logger.error("  TAVILY SEARCH ERROR for query[%d] after %.2fs: %s", i, elapsed, e)
            # Append empty result to maintain indexing
            search_docs.append({'results': []})

    logger.debug("TAVILY_SEARCH_MULTIPLE COMPLETE | total_results=%d from %d queries", 
                total_results, len(search_queries))
    return search_docs

def summarize_webpage_content(webpage_content: str) -> str:
    """Summarize webpage content using the configured summarization model.

    Args:
        webpage_content: Raw webpage content to summarize

    Returns:
        Formatted summary with key excerpts
    """
    content_len = len(webpage_content)
    logger.debug("SUMMARIZE_WEBPAGE | input_len=%d chars", content_len)
    
    start_time = time.perf_counter()
    try:
        # Set up structured output model for summarization
        structured_model = summarization_model.with_structured_output(Summary)

        # Generate summary
        summary = structured_model.invoke([
            HumanMessage(content=summarize_webpage_prompt.format(
                webpage_content=webpage_content, 
                date=get_today_str()
            ))
        ])
        
        elapsed = time.perf_counter() - start_time

        # Format summary with clear structure
        formatted_summary = (
            f"<summary>\n{summary.summary}\n</summary>\n\n"
            f"<key_excerpts>\n{summary.key_excerpts}\n</key_excerpts>"
        )
        
        logger.debug("SUMMARIZE_WEBPAGE COMPLETE | input=%d -> output=%d chars in %.2fs", 
                    content_len, len(formatted_summary), elapsed)

        return formatted_summary

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.warning("SUMMARIZE_WEBPAGE FAILED after %.2fs: %s (returning truncated content)", 
                      elapsed, str(e))
        return webpage_content[:2000] + "..." if len(webpage_content) > 2000 else webpage_content

def deduplicate_search_results(search_results: List[dict]) -> dict:
    """Deduplicate search results by URL to avoid processing duplicate content.

    Args:
        search_results: List of search result dictionaries

    Returns:
        Dictionary mapping URLs to unique results
    """
    unique_results = {}
    total_results = 0
    duplicates = 0

    for response in search_results:
        for result in response.get('results', []):
            total_results += 1
            url = result['url']
            if url not in unique_results:
                unique_results[url] = result
            else:
                duplicates += 1

    logger.debug("DEDUPLICATE | total=%d, unique=%d, duplicates=%d", 
                total_results, len(unique_results), duplicates)
    return unique_results

def process_search_results(unique_results: dict) -> dict:
    """Process search results by summarizing content where available.

    Args:
        unique_results: Dictionary of unique search results

    Returns:
        Dictionary of processed results with summaries
    """
    logger.debug("PROCESS_SEARCH_RESULTS | processing %d unique results", len(unique_results))
    
    summarized_results = {}
    summarized_count = 0
    fallback_count = 0
    total_start = time.perf_counter()

    for i, (url, result) in enumerate(unique_results.items()):
        title = result.get('title', 'Untitled')[:50]
        
        # Use existing content if no raw content for summarization
        if not result.get("raw_content"):
            content = result['content']
            fallback_count += 1
            logger.debug("  result[%d] '%s': using existing content (%d chars)", 
                        i, title, len(content))
        else:
            # Summarize raw content for better processing
            raw_len = len(result['raw_content'])
            truncated_len = min(raw_len, MAX_CONTEXT_LENGTH)
            logger.debug("  result[%d] '%s': summarizing raw content (%d -> %d chars)", 
                        i, title, raw_len, truncated_len)
            content = summarize_webpage_content(result['raw_content'][:MAX_CONTEXT_LENGTH])
            summarized_count += 1

        summarized_results[url] = {
            'title': result['title'],
            'content': content
        }

    elapsed = time.perf_counter() - total_start
    logger.debug("PROCESS_SEARCH_RESULTS COMPLETE | summarized=%d, fallback=%d, total_time=%.2fs",
                summarized_count, fallback_count, elapsed)
    return summarized_results

def format_search_output(summarized_results: dict) -> str:
    """Format search results into a well-structured string output.

    Args:
        summarized_results: Dictionary of processed search results

    Returns:
        Formatted string of search results with clear source separation
    """
    if not summarized_results:
        return "No valid search results found. Please try different search queries or use a different search API."

    formatted_output = "Search results: \n\n"

    for i, (url, result) in enumerate(summarized_results.items(), 1):
        formatted_output += f"\n\n--- SOURCE {i}: {result['title']} ---\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        formatted_output += "-" * 80 + "\n"

    return formatted_output

# ===== RESEARCH TOOLS =====

@tool(parse_docstring=True)
def tavily_search(
    query: str,
    max_results: Annotated[int, InjectedToolArg] = 3,
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
) -> str:
    """Performs a web search using the Tavily API with content summarization.

    Returns a summarized string of the top search results.

    Query Guidelines:
    - `query` must be a **single** focused search topic or question. Max 50 words.
    Good: "What are the latest FDA guidelines for sterile compounding?"
    Bad: "sterile compounding guidelines pharmacy regulations" (Keywords are less effective for summarization)
    Bad: "Who is the CEO of Pfizer? And what is their stock price?" (Multiple questions)

    Args:
        query: What information you are looking for (expressed as a concise single question)
        max_results: Maximum number of results to return
        topic: Topic to filter results by ('general', 'news', 'finance')

    Returns:
        Formatted string of search results with summaries
    """
    query_preview = query[:80] + "..." if len(query) > 80 else query
    logger.info("TAVILY_SEARCH_TOOL | query='%s'", query_preview)
    
    start_time = time.perf_counter()
    
    try:
        # Execute search for single query
        search_results = tavily_search_multiple(
            [query],  # Convert single query to list for the internal function
            max_results=max_results,
            topic=topic,
            include_raw_content=True,
        )

        # Deduplicate results by URL to avoid processing duplicate content
        unique_results = deduplicate_search_results(search_results)

        # Process results with summarization
        summarized_results = process_search_results(unique_results)

        # Format output for consumption
        output = format_search_output(summarized_results)
        
        elapsed = time.perf_counter() - start_time
        logger.info("TAVILY_SEARCH_TOOL COMPLETE | results=%d, output_len=%d chars, time=%.2fs",
                   len(summarized_results), len(output), elapsed)
        
        return output
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error("TAVILY_SEARCH_TOOL ERROR after %.2fs: %s", elapsed, e)
        raise

@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Strategic reasoning engine to validate research progress and determine termination.

    Use this tool to enforce the "Article First, Web Second" protocol and audit draft quality.

    Required Reflection Structure:
    1. CURRENT INVENTORY (The "VP Test"): 
       - Does the draft contain specific NUMBERS (n=, p-values, costs) from the User Article?
       - Are there still placeholders (e.g., "[Insert Result]") that block publication?
    2. BLIND SPOTS: 
       - What would a skeptical Pharmacy Director ask that I cannot answer yet?
       - Is critical external context (guidelines, safety signals) missing?
    3. SOURCE HYGIENE: 
       - Have I strictly finished Phase 1 (Internal extraction) before starting Phase 2 (External context)?
    4. STRATEGIC DECISION:
       - If User Article data is missing -> Call ConductResearch(search_type="internal")
       - If Article is done but Context is missing -> Call ConductResearch(search_type="external")
       - If Draft is factually dense, accurate, and free of placeholders -> Call ResearchComplete

    Args:
        reflection: A structural audit of the draft's completeness, source priority, and readiness for publication.

    Returns:
        Minimal acknowledgment (reflection content is already captured in the tool call)
    """
    # Return minimal acknowledgment - the reflection content is already captured
    # in the AIMessage's tool_call arguments, so we don't need to echo it back
    return "Acknowledged."

@tool(parse_docstring=True)
def refine_draft_report(
    research_brief: Annotated[str, InjectedToolArg],
    article_summary: Annotated[str, InjectedToolArg],
    findings: Annotated[str, InjectedToolArg],
    draft_report: Annotated[str, InjectedToolArg],
    newsletter_template: Annotated[str, InjectedToolArg],
):
    """Refine draft report

    Synthesizes all research findings into a comprehensive draft report

    Args:
        research_brief: concise user/editorial intent to preserve while refining
        article_summary: summary of the article being processed
        findings: collected research findings for the user request
        draft_report: draft report based on the findings and user request
        newsletter_template: selected template type (research_article or commentary)

    Returns:
        refined draft report
    """
    logger.info("="*60)
    logger.info("REFINE_DRAFT_REPORT START")
    logger.info("  research_brief_len: %d chars", len(research_brief) if research_brief else 0)
    logger.info("  article_summary_len: %d chars", len(article_summary) if article_summary else 0)
    logger.info("  findings_len: %d chars", len(findings) if findings else 0)
    logger.info("  draft_report_len: %d chars", len(draft_report) if draft_report else 0)
    logger.info("  newsletter_template: %s", newsletter_template or "commentary")
    
    start_time = time.perf_counter()
    
    # Validate newsletter_template - fail fast if not set
    if not newsletter_template:
        raise ValueError("newsletter_template not provided to refine_draft_report. This should be passed from supervisor state.")
    
    # Get formatted template for prompt injection
    template_content = format_template_for_prompt(newsletter_template)

    try:
        draft_report_prompt = report_generation_with_draft_insight_prompt.format(
            research_brief=research_brief,
            article_summary=article_summary,
            findings=findings,
            draft_report=draft_report,
            date=get_today_str(),
            template_content=template_content
        )
        
        logger.debug("  prompt_len: %d chars", len(draft_report_prompt))

        response = writer_model.invoke([HumanMessage(content=draft_report_prompt)])
        
        elapsed = time.perf_counter() - start_time
        output_len = len(response.content) if response.content else 0
        
        # Log token usage
        log_token_usage(logger, response, "refine_draft_report")
        get_global_tracker().add_usage(response, "refine_draft_report")
        
        logger.info("REFINE_DRAFT_REPORT COMPLETE | output_len=%d chars, time=%.2fs", 
                   output_len, elapsed)
        
        return response.content
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error("REFINE_DRAFT_REPORT ERROR after %.2fs: %s", elapsed, e)
        raise


# ===== DOCUMENT INGESTION AND RETRIEVAL =====

def _read_pdf_text(pdf_path: Path, config: Optional[PdfProcessorConfig] = None) -> str:
    """Extract PDF text via docling VLM with DocTags prompt (fallback to standard docling)."""
    cfg = config or PDF_PROCESSOR_CONFIG

    cache_path: Optional[Path] = None
    if CACHE_ENABLED:
        try:
            pdf_hash = _sha256_bytes(pdf_path.read_bytes())
            cache_path = CACHE_DIR / "pdf_ocr" / f"{pdf_hash}_{INGEST_CACHE_VERSION}" / "extracted.md"
            if cache_path.exists():
                cached = cache_path.read_text(encoding="utf-8")
                if cached.strip():
                    logger.info("PDF OCR cache hit: %s", cache_path)
                    return cached
        except Exception as exc:  # pragma: no cover - cache best-effort
            logger.debug("PDF OCR cache read failed for %s: %s", pdf_path, exc)

    try:
        markdown = extract_pdf_to_markdown(pdf_path, cfg)
        if cache_path and markdown and markdown.strip():
            try:
                _write_text_atomic(cache_path, markdown)
                logger.info("PDF OCR cached: %s", cache_path)
            except Exception as exc:  # pragma: no cover - cache best-effort
                logger.debug("PDF OCR cache write failed for %s: %s", cache_path, exc)
        return markdown
    except PdfDependencyError as exc:
        logger.error("Missing dependency for PDF extraction at %s: %s", pdf_path, exc)
    except Exception as exc:
        logger.warning("Failed to read PDF at %s: %s", pdf_path, exc)
    return ""

def _extract_text_from_html(html: str) -> str:
    """Best-effort HTML -> text extraction with optional dependencies."""
    if not html or not html.strip():
        return ""

    # Prefer high-quality extractors when available (optional deps).
    try:  # pragma: no cover - optional dependency
        import trafilatura  # type: ignore

        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True,
            include_links=False,
        )
        if extracted and extracted.strip():
            return extracted.strip()
    except Exception:
        pass

    try:  # pragma: no cover - optional dependency
        from bs4 import BeautifulSoup  # type: ignore

        try:
            soup = BeautifulSoup(html, "lxml")  # type: ignore[arg-type]
        except Exception:
            soup = BeautifulSoup(html, "html.parser")  # type: ignore[arg-type]
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = soup.get_text("\n")
        text = html_lib.unescape(text)
        # Collapse whitespace a bit
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()
    except Exception:
        pass

    # Fallback: extremely naive tag stripping (keeps readability reasonable for many pages).
    cleaned = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", "", html)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = html_lib.unescape(cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def fetch_url_as_text(url: str, timeout_s: float = 20.0, max_bytes: int = 5_000_000) -> str:
    """Fetch a URL and extract readable text (best-effort).

    This is used for URL-based ingestion so the rest of the pipeline can treat it
    like pasted text/PDF content. Caches extracted text when enabled.
    """
    if not url or not str(url).strip():
        return ""

    url = str(url).strip()

    cache_path: Optional[Path] = None
    if CACHE_ENABLED:
        try:
            url_hash = _sha256_text(url)
            cache_path = CACHE_DIR / "url_text" / f"{url_hash}_{INGEST_CACHE_VERSION}" / "extracted.txt"
            if cache_path.exists():
                cached = cache_path.read_text(encoding="utf-8")
                if cached.strip():
                    logger.info("URL text cache hit: %s", cache_path)
                    return cached
        except Exception as exc:  # pragma: no cover - cache best-effort
            logger.debug("URL cache read failed for %s: %s", url, exc)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            content_type = str(resp.headers.get("Content-Type", "") or "")
            if "application/pdf" in content_type.lower():
                logger.warning("URL appears to be a PDF (%s); provide a pdf_path instead: %s", content_type, url)
                return ""

            data = resp.read(max_bytes + 1)
            if len(data) > max_bytes:
                logger.warning("URL response too large (> %d bytes): %s", max_bytes, url)
                return ""

            charset = None
            try:
                charset = resp.headers.get_content_charset()
            except Exception:
                charset = None
            html = data.decode(charset or "utf-8", errors="ignore")
    except urllib.error.URLError as exc:
        logger.warning("Failed to fetch URL %s: %s", url, exc)
        return ""
    except Exception as exc:
        logger.warning("Unexpected error fetching URL %s: %s", url, exc)
        return ""

    extracted = _extract_text_from_html(html)
    if cache_path and extracted and extracted.strip():
        try:
            _write_text_atomic(cache_path, extracted)
            logger.info("URL text cached: %s", cache_path)
        except Exception as exc:  # pragma: no cover - cache best-effort
            logger.debug("URL cache write failed for %s: %s", url, exc)

    return extracted


def generate_article_summary(text: str, max_words: int = 500) -> str:
    """Generate a global summary of the article for use as context throughout the pipeline.
    
    This summary provides high-level understanding of the article to help with:
    - Planning which questions to ask
    - Understanding what to look for in chunk retrieval
    - Providing context for research agents
    
    Args:
        text: Full article text
        max_words: Target word count for the summary (default 500)
        
    Returns:
        A concise summary of the article, or empty string on failure
    """
    if not text or len(text.strip()) < 100:
        logger.warning("Text too short for summary generation")
        return ""
    
    # Truncate if extremely long to fit in context window
    max_chars = 120000  # ~30k tokens
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Text truncated for summary generation...]"
    
    summary_prompt = article_summary_prompt.format(max_words=max_words, text=text)

    logger.info("Generating global article summary (~%d words)...", max_words)
    start_time = time.perf_counter()
    
    try:
        response = article_summary_model.invoke([HumanMessage(content=summary_prompt)])
        elapsed = time.perf_counter() - start_time
        
        summary = response.content if response.content else ""
        word_count = len(summary.split())
        
        logger.info("Generated article summary: %d words, %d chars in %.2fs", 
                   word_count, len(summary), elapsed)
        
        return summary
        
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        logger.error("Failed to generate article summary after %.2fs: %s", elapsed, e)
        return ""


def _chunk_text(text: str, config: Optional[PdfProcessorConfig] = None) -> List[Document]:
    """Token-based, table-aware chunking wrapper around pdf_processor.chunk_text."""
    if not text or not text.strip():
        return []

    cfg = config or PDF_PROCESSOR_CONFIG
    try:
        chunks = chunk_pdf_text(text, cfg)
        filtered_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]
        return filtered_chunks
    except PdfDependencyError as exc:
        logger.error("Missing dependency during chunking: %s", exc)
        raise RuntimeError(f"Chunking dependencies missing: {exc}") from exc
    except Exception as exc:
        logger.error("Chunking failed: %s", exc)
        raise RuntimeError(f"Chunking failed: {exc}") from exc


def store_document(doc_id: str, text: str, generate_summary: bool = True) -> Tuple[int, int, str]:
    """Store a text document in an in-memory chunk store with a global summary.

    Chunks are later reranked with a cross-encoder (no vector search).

    Args:
        doc_id: Identifier for the document
        text: Full text content to chunk
        generate_summary: Whether to generate a global summary (default True)

    Returns:
        Tuple of (chunk_count, total_chars, global_summary)
    """
    cache_dir: Optional[Path] = None
    chunks_cache_path: Optional[Path] = None
    summary_cache_path: Optional[Path] = None
    if CACHE_ENABLED and text and text.strip():
        try:
            text_hash = _sha256_text(text)
            cache_dir = CACHE_DIR / "documents" / doc_id / f"{text_hash}_{INGEST_CACHE_VERSION}"
            chunks_cache_path = cache_dir / "chunks.json"
            summary_cache_path = cache_dir / "global_summary.txt"

            if chunks_cache_path.exists():
                try:
                    cached_chunks = _read_json(chunks_cache_path)
                    if isinstance(cached_chunks, list) and cached_chunks:
                        chunks = [
                            Document(
                                page_content=str(item.get("page_content", "")),
                                metadata=dict(item.get("metadata", {})),
                            )
                            for item in cached_chunks
                            if isinstance(item, dict) and item.get("page_content")
                        ]
                        if chunks:
                            global_summary = ""
                            if summary_cache_path.exists():
                                global_summary = summary_cache_path.read_text(encoding="utf-8")
                            logger.info(
                                "Document cache hit: chunks=%d, summary=%s (%s)",
                                len(chunks),
                                "yes" if bool(global_summary.strip()) else "no",
                                cache_dir,
                            )
                            # Attach stable locators even for cached chunks (backfill if needed)
                            for idx, chunk in enumerate(chunks):
                                chunk.metadata.setdefault("doc_id", doc_id)
                                chunk.metadata.setdefault("chunk_index", idx)
                                chunk.metadata.setdefault("chunk_id", f"{doc_id}:{idx}")
                            # If summary is missing and requested, compute and cache it.
                            if generate_summary and not global_summary.strip() and len(text.strip()) > 100:
                                global_summary = generate_article_summary(text)
                                if global_summary.strip():
                                    try:
                                        _write_text_atomic(summary_cache_path, global_summary)
                                    except Exception as exc:  # pragma: no cover
                                        logger.debug("Summary cache write failed for %s: %s", summary_cache_path, exc)
                            # Store and return from cache
                            total_chars = sum(len(c.page_content) for c in chunks)
                            with _STORE_LOCK:
                                DOCUMENT_CHUNKS[doc_id] = chunks
                                DOCUMENT_METADATA[doc_id] = {"global_summary": global_summary}
                            return len(chunks), total_chars, global_summary
                except Exception as exc:  # pragma: no cover - cache best-effort
                    logger.debug("Chunk cache read failed for %s: %s", chunks_cache_path, exc)
        except Exception as exc:  # pragma: no cover - cache best-effort
            logger.debug("Document cache setup failed: %s", exc)

    # Generate global summary first (before chunking) for use as context
    global_summary = ""
    if generate_summary and text and len(text.strip()) > 100:
        global_summary = generate_article_summary(text)

    chunks = _chunk_text(text)
    if not chunks:
        with _STORE_LOCK:
            DOCUMENT_CHUNKS[doc_id] = []
            DOCUMENT_METADATA[doc_id] = {"global_summary": global_summary}
        logger.info("Document '%s' produced no chunks (empty or whitespace-only text)", doc_id)
        return 0, 0, global_summary

    # Attach stable locators for downstream traceability (e.g., evidence ledger).
    # These are stable within a run and stable across reruns when OCR+chunking is unchanged.
    for idx, chunk in enumerate(chunks):
        chunk.metadata.setdefault("doc_id", doc_id)
        chunk.metadata.setdefault("chunk_index", idx)
        chunk.metadata.setdefault("chunk_id", f"{doc_id}:{idx}")

    total_chars = sum(len(c.page_content) for c in chunks)
    logger.info(
        "Chunked document '%s' into %d chunks (%d chars total)",
        doc_id,
        len(chunks),
        total_chars,
    )

    with _STORE_LOCK:
        DOCUMENT_CHUNKS[doc_id] = chunks
        DOCUMENT_METADATA[doc_id] = {"global_summary": global_summary}

    # Best-effort persistent cache write (OCR output may already be cached separately).
    if cache_dir and chunks_cache_path:
        try:
            serializable_chunks = [
                {"page_content": c.page_content, "metadata": dict(c.metadata)} for c in chunks
            ]
            _write_json_atomic(chunks_cache_path, serializable_chunks)
            if summary_cache_path and global_summary.strip():
                _write_text_atomic(summary_cache_path, global_summary)
            logger.info("Document cached: %s", cache_dir)
        except Exception as exc:  # pragma: no cover - cache best-effort
            logger.debug("Document cache write failed for %s: %s", cache_dir, exc)

    logger.info(
        "Stored document '%s': %d chunks (%d chars) in in-memory store, summary=%d words",
        doc_id,
        len(chunks),
        total_chars,
        len(global_summary.split()),
    )
    return len(chunks), total_chars, global_summary


@tool(parse_docstring=True)
def ingest_document(doc_id: str, pdf_path: str = "", text: str = "") -> dict:
    """Ingest a PDF or raw text into an in-memory chunk store for later retrieval.

    Args:
        doc_id: Identifier used later to retrieve chunks (e.g., 'user_article')
        pdf_path: Path to the PDF file to ingest (optional)
        text: Raw text to ingest (optional)

    Returns:
        Dict with stats about stored chunks, global summary, or an error message.
    """
    if not pdf_path and not text:
        return {"status": "error", "message": "Provide either pdf_path or text."}

    content = text
    if pdf_path:
        path = Path(pdf_path)
        if not path.exists():
            return {"status": "error", "message": f"PDF not found at {pdf_path}"}
        content = _read_pdf_text(path)
        if not content:
            return {"status": "error", "message": f"Unable to read PDF at {pdf_path}"}

    try:
        chunk_count, total_chars, global_summary = store_document(doc_id, content)
        return {
            "status": "success",
            "message": f"Ingested document '{doc_id}' with {chunk_count} chunks ({total_chars} chars).",
            "doc_id": doc_id,
            "chunk_count": chunk_count,
            "total_chars": total_chars,
            "global_summary": global_summary,
            "article_content": content  # Full raw content for state storage
        }
    except RuntimeError as e:
        return {"status": "error", "message": f"ERROR: {e}"}


@tool(parse_docstring=True)
def retrieve_document_chunks(doc_id: str, query: str) -> str:
    """Retrieves relevant chunks using cross-encoder reranking.

    Returns the top 5 chunks joined by separators.

    Query Guidelines:
    - `query` must be a **single** natural-language question. Max 50 words.
    Good: "How were falls reported in the study?"
    Bad: "What is the cohort size? Also, what is the date range?" (Multiple questions)
    Bad: "cohort size date range" (Keywords)

    Args:
        doc_id: Document identifier used during ingest (e.g., 'user_article')
        query: What information you are looking for (expressed as a concise single question)

    Returns:
        Top reranked chunks joined with separators. If the doc is missing, returns a warning.
        Returns a message if no chunks meet the threshold.
    """
    score_threshold = 0.20  # 0.20 to reduce false positives and preserve context for relevant material
    
    if not query or not query.strip():
        return "Query must be a non-empty string."

    # Thread-safe read from chunk store
    with _STORE_LOCK:
        chunks = DOCUMENT_CHUNKS.get(doc_id)

    if chunks is None:
        return f"No document found for doc_id '{doc_id}'. Call ingest_document first."

    if not chunks:
        return f"Document '{doc_id}' has no chunks stored."

    try:
        reranker = _get_reranker()
    except Exception as exc:
        logger.error("Reranker initialization failed: %s", exc)
        return f"Error during reranking: {exc}"

    pairs = [[query, c.page_content] for c in chunks]
    logger.info(
        "RERANKING START | doc_id=%s | query_len=%d | chunk_count=%d | threshold=%.2f",
        doc_id,
        len(query),
        len(chunks),
        score_threshold,
    )

    try:
        scores = _compute_reranker_scores(reranker, pairs)
    except Exception as exc:
        if _is_meta_tensor_device_error(exc):
            logger.warning(
                "Meta-tensor reranker failure for doc '%s'; resetting cached reranker and retrying on CPU once: %s",
                doc_id,
                exc,
            )
            _reset_reranker()
            try:
                reranker = _get_reranker(force_device="cpu")
                scores = _compute_reranker_scores(reranker, pairs)
                logger.warning(
                    "Reranker recovery succeeded for doc '%s' after reloading on CPU",
                    doc_id,
                )
            except Exception as retry_exc:
                logger.error(
                    "Error computing reranker scores for doc '%s' after CPU recovery attempt: %s",
                    doc_id,
                    retry_exc,
                )
                return f"Error during reranking: {retry_exc}"
        else:
            logger.error("Error computing reranker scores for doc '%s': %s", doc_id, exc)
            return f"Error during reranking: {exc}"

    scores_list, non_finite_count = _normalize_reranker_scores(scores)
    if non_finite_count:
        logger.warning(
            "Reranker returned %d non-finite scores for doc '%s'; coercing them to 0.0",
            non_finite_count,
            doc_id,
        )
        if non_finite_count == len(scores_list):
            return (
                "Error during reranking: non-finite scores returned by the document reranker. "
                "This indicates a retrieval-tool failure, not necessarily missing content in the document."
            )

    if len(scores_list) != len(chunks):
        logger.warning(
            "Reranker returned %d scores for %d chunks; truncating to min length",
            len(scores_list),
            len(chunks),
        )
        min_len = min(len(scores_list), len(chunks))
        scores_list = scores_list[:min_len]
        chunks = chunks[:min_len]

    # Zip chunks and scores together to keep them paired
    scored_chunks = list(zip(chunks, scores_list))

    # Filter based on the threshold - removes poor results immediately
    filtered_results = [
        (chunk, score) for chunk, score in scored_chunks 
        if score >= score_threshold
    ]

    if not filtered_results:
        # Find the highest score that was filtered out
        max_score = max(scores_list) if scores_list else 0.0
        
        logger.info(
            "RERANKING COMPLETE | doc_id=%s | returned=0 | filtered_out=%d | threshold=%.2f | max_score=%.3f",
            doc_id,
            len(chunks),
            score_threshold,
            max_score,
        )
        
        if max_score > 0:
            # Identify the specific chunk that was the "closest loser" to provide context
            best_idx = scores_list.index(max_score)
            best_chunk = chunks[best_idx]
            best_text = best_chunk.page_content[:200].replace("\n", " ") + "..."
            best_chunk_id = best_chunk.metadata.get("chunk_id")
            best_page = best_chunk.metadata.get("page")
            best_locator_parts = []
            if best_chunk_id:
                best_locator_parts.append(f"id={best_chunk_id}")
            if best_page:
                best_locator_parts.append(f"page={best_page}")
            best_locator = " | ".join(best_locator_parts) if best_locator_parts else "id=(unknown)"

            return (
                f"No chunks met confidence threshold (Threshold: {score_threshold}, Best: {max_score:.3f}).\n"
                f"Closest match locator: {best_locator}\n"
                f"Closest match text: \"{best_text}\"\n"
                f"Hint: Compare your query to the closest match. If relevant, retry using specific terms found in the text. If unrelated, the document likely lacks this info."
            )
        else:
            return (
                f"No relevant chunks found (Max Score: 0.0). "
                f"The document does not appear to contain information semantically related to this query."
            )

    # Sort by score descending
    filtered_results.sort(key=lambda x: x[1], reverse=True)

    # Take top 5 (or fewer if we filtered many out)
    top_results = filtered_results[:5]

    formatted: List[str] = []
    for rank, (ch, score) in enumerate(top_results, start=1):
        block_type = ch.metadata.get("block_type", "text")
        chunk_id = ch.metadata.get("chunk_id")
        page = ch.metadata.get("page")
        block_index = ch.metadata.get("block_index")
        locator_parts = []
        if chunk_id:
            locator_parts.append(f"id={chunk_id}")
        if page:
            locator_parts.append(f"page={page}")
        if block_index is not None:
            locator_parts.append(f"block={block_index}")
        locator = " | ".join(locator_parts) if locator_parts else "id=(unknown)"
        formatted.append(
            f"[chunk {rank}] {locator} | score={score:.3f} | type={block_type}\n{ch.page_content}"
        )

    logger.info(
        "RERANKING COMPLETE | doc_id=%s | returned=%d | filtered_out=%d | threshold=%.2f",
        doc_id,
        len(top_results),
        len(chunks) - len(filtered_results),
        score_threshold,
    )

    return "\n\n---\n\n".join(formatted)


# ===== NEWSLETTER PARSING =====

# Note: 're' is imported at top of file
from src.state_scope import StructuredNewsletter, NewsletterSection, NewsletterSource


def _sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to be safe for file system use.
    
    Removes or replaces characters that could cause issues:
    - Path separators (/, \\)
    - Special characters (<, >, :, ", |, ?, *)
    - Leading/trailing whitespace and dots
    - Replaces spaces with underscores
    """
    # Remove path separators and dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Remove leading/trailing whitespace and dots
    sanitized = sanitized.strip('. \t\n\r')
    # Collapse multiple underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    # Fallback if empty
    if not sanitized:
        sanitized = "newsletter"
    return sanitized


_SOURCE_LINE_PATTERN = re.compile(r"^\s*(?:(?:[-*]|\d+\.)\s*)?\[(\d+)\]\s*(.+?)\s*$")
_MARKDOWN_LINK_PATTERN = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_URL_PATTERN = re.compile(r"https?://[^\s<>\]]+")


def _extract_urls_from_text(text: str) -> list[str]:
    """Extract URLs from plain text or markdown links in a stable order."""
    urls: list[str] = []

    for match in _MARKDOWN_LINK_PATTERN.finditer(text):
        url = match.group(2).rstrip(").,;")
        if url and url not in urls:
            urls.append(url)

    for match in _URL_PATTERN.finditer(text):
        url = match.group(0).rstrip(").,;")
        if url and url not in urls:
            urls.append(url)

    return urls


def _clean_source_title(source_text: str, url: str | None) -> str:
    """Remove URL syntax and trailing punctuation from a source line."""
    if not source_text:
        return ""

    title_text = _MARKDOWN_LINK_PATTERN.sub(lambda m: m.group(1), source_text)
    if url:
        title_text = title_text.replace(url, " ")
    title_text = re.sub(r"\(\s*\)", " ", title_text)
    title_text = re.sub(r"\s*[:\-\u2013\u2014]\s*$", "", title_text)
    title_text = re.sub(r"\s+", " ", title_text).strip(" -:\t")

    if title_text:
        return title_text
    if url:
        parsed = urlparse(url)
        return parsed.netloc or url
    return ""


def _parse_sources_block(sources_text: str) -> list[NewsletterSource]:
    """Parse a markdown sources block using tolerant line patterns."""
    sources: list[NewsletterSource] = []

    for raw_line in sources_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _SOURCE_LINE_PATTERN.match(line)
        if not match:
            continue

        source_id = int(match.group(1))
        source_text = match.group(2).strip()
        urls = _extract_urls_from_text(source_text)
        url = urls[0] if urls else None
        title_text = _clean_source_title(source_text, url)

        if not title_text:
            title_text = url or f"Source {source_id}"

        sources.append(
            NewsletterSource(
                id=source_id,
                title=title_text,
                url=url,
            )
        )

    return sources


def sources_from_evidence_ledger(
    evidence_ledger: list[dict],
    *,
    include_article_sources: bool = True,
) -> list[dict]:
    """Derive a minimal sources array from the structured evidence ledger."""
    derived_sources: list[dict] = []
    seen_keys: set[str] = set()
    next_id = 1

    for entry in evidence_ledger or []:
        for evidence in entry.get("evidence", []) or []:
            source_type = str(evidence.get("source_type", "") or "").strip()
            locator = str(evidence.get("locator", "") or "").strip()
            title = str(evidence.get("title", "") or "").strip()

            if source_type == "article_chunk":
                if not include_article_sources:
                    continue
                key = "provided_article"
                title = title or "Provided article (user-supplied text)"
                url = None
            else:
                url = locator if locator.startswith("http") else None
                key = url or title
                if not key:
                    continue
                if not title:
                    title = urlparse(url).netloc if url else "External source"

            if key in seen_keys:
                continue

            seen_keys.add(key)
            derived_sources.append({"id": next_id, "title": title, "url": url})
            next_id += 1

    return derived_sources


def parse_newsletter_to_structured(
    markdown: str, 
    newsletter_template: str = "commentary"
) -> StructuredNewsletter:
    """
    Parse a markdown newsletter into a structured format for website import.
    
    Works for both research_article and commentary newsletter structures.
    
    Args:
        markdown: The markdown newsletter content
        newsletter_template: The template type for tagging (e.g., 'research_article' or 'commentary')
    
    Returns:
        StructuredNewsletter object with parsed sections
    """
    logger.info("PARSE_NEWSLETTER_TO_STRUCTURED | input_len=%d chars", len(markdown))
    
    # Validate newsletter_template - fail fast if not set
    if not newsletter_template:
        raise ValueError("newsletter_template not provided to parse_newsletter_to_structured.")
    
    # Use the template directly for tagging
    newsletter_structure = newsletter_template
    
    logger.info("  newsletter_structure: %s", newsletter_structure)

    def _is_placeholder_title_line(line: str) -> bool:
        return bool(_PLACEHOLDER_TITLE_RE.match(str(line or "").strip()))

    def _extract_title_after_placeholder(lines: list[str], heading_idx: int) -> str:
        in_metadata_block = False
        for line in lines[heading_idx + 1 :]:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == _SOURCE_METADATA_START:
                in_metadata_block = True
                continue
            if stripped == _SOURCE_METADATA_END:
                in_metadata_block = False
                continue
            if in_metadata_block or stripped.startswith("<!--"):
                continue
            if stripped.startswith("## "):
                break
            if _is_placeholder_title_line(stripped):
                continue
            bullet_match = re.match(r"^(?:[-*]|\d+[.)])\s+(.+)$", stripped)
            if bullet_match:
                candidate = bullet_match.group(1).strip()
                if candidate and not _is_placeholder_title_line(candidate):
                    return candidate
                continue
            if not stripped.startswith("#"):
                return stripped
        return ""
    
    # Extract title (# heading)
    title = ""
    title_match = re.search(r'^#\s+(.+?)(?:\n|$)', markdown, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
    else:
        for line in markdown.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("## "):
                break
            title = re.sub(r"^#+\s*", "", stripped)
            break

    if _is_placeholder_title_line(title):
        lines = markdown.splitlines()
        heading_idx = None
        for idx, line in enumerate(lines):
            if re.match(r"^#\s+t?itle\s*$", line.strip(), re.IGNORECASE):
                heading_idx = idx
                break
        if heading_idx is not None:
            title = _extract_title_after_placeholder(lines, heading_idx)

    logger.info("  extracted_title: %s", title[:50] + "..." if len(title) > 50 else title)
    
    # Extract sections (## headings)
    sections = []
    
    # Split by ## headings, keeping the heading names
    section_pattern = r'^##\s+(.+?)(?:\n|$)'
    section_matches = list(re.finditer(section_pattern, markdown, re.MULTILINE))
    
    for i, match in enumerate(section_matches):
        section_name = match.group(1).strip()
        start_pos = match.end()
        
        # Find end position (next ## or end of document, or ## Sources)
        if i + 1 < len(section_matches):
            end_pos = section_matches[i + 1].start()
        else:
            end_pos = len(markdown)
        
        # Extract content between this heading and the next
        content = markdown[start_pos:end_pos].strip()
        
        # Skip the Sources section - we'll parse it separately
        if section_name.lower() == "sources":
            continue
        
        sections.append(NewsletterSection(
            name=section_name,
            content=content
        ))
    
    logger.info("  extracted_sections: %d", len(sections))
    for s in sections:
        logger.debug("    - %s (%d chars)", s.name, len(s.content))
    
    # Extract sources
    sources = []
    sources_match = re.search(r'^##\s+Sources?\s*\n(.*?)(?=^##|\Z)', markdown, re.MULTILINE | re.DOTALL)
    if sources_match:
        sources_text = sources_match.group(1).strip()
        sources = _parse_sources_block(sources_text)
    
    logger.info("  extracted_sources: %d", len(sources))
    
    result = StructuredNewsletter(
        newsletter_structure=newsletter_structure,
        title=title,
        sections=sections,
        sources=sources,
        raw_markdown=markdown
    )
    
    logger.info("PARSE_NEWSLETTER_TO_STRUCTURED COMPLETE")
    
    return result


def structured_newsletter_to_dict(newsletter: StructuredNewsletter) -> dict:
    """
    Convert a StructuredNewsletter to a plain dictionary for JSON serialization.
    
    Args:
        newsletter: The StructuredNewsletter object
    
    Returns:
        Dictionary representation suitable for JSON export
    """
    return {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "version": "1.0"
        },
        "newsletter_structure": newsletter.newsletter_structure,
        "title": newsletter.title,
        "sections": [
            {"name": s.name, "content": s.content}
            for s in newsletter.sections
        ],
        "sources": [
            {"id": s.id, "title": s.title, "url": s.url}
            for s in newsletter.sources
        ],
        "raw_markdown": newsletter.raw_markdown
    }
