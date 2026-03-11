"""
PDF extraction and chunking utilities for the newsletter pipeline.

This module uses OpenAI's vision models as a PDF OCR engine to produce clean
Markdown, then applies a table-aware, token-based chunker adapted from
`docling_granite.py`.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.logging_config import get_logger, log_timing

logger = get_logger("deep_research.pdf_processor")

# ===== REQUIRED DEPENDENCIES =====

from pdf2image import convert_from_path
from PIL import Image
from openai import OpenAI
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ===== DEFAULTS =====

from src.model_config import PIPELINE_MODEL_SETTINGS
from src.prompts import ocr_extraction_prompt

DEFAULT_EXTRACTION_PROMPT = ocr_extraction_prompt

DEFAULT_OCR_MODEL = PIPELINE_MODEL_SETTINGS.pdf_ocr_transcription_model.model_name
DEFAULT_OCR_DPI = PIPELINE_MODEL_SETTINGS.pdf_ocr_transcription_model.dpi
DEFAULT_OCR_MAX_IMAGE_DIMENSION = PIPELINE_MODEL_SETTINGS.pdf_ocr_transcription_model.max_image_dimension
DEFAULT_OCR_MAX_OUTPUT_TOKENS = PIPELINE_MODEL_SETTINGS.pdf_ocr_transcription_model.max_output_tokens
DEFAULT_OCR_CONCURRENCY = PIPELINE_MODEL_SETTINGS.pdf_ocr_transcription_model.concurrency
DEFAULT_OCR_TEMPERATURE = PIPELINE_MODEL_SETTINGS.pdf_ocr_transcription_model.temperature
DEFAULT_OCR_REASONING_EFFORT = PIPELINE_MODEL_SETTINGS.pdf_ocr_transcription_model.reasoning_effort

DEFAULT_TOKEN_MODEL = PIPELINE_MODEL_SETTINGS.pdf_chunk_tokenizer_model.model_name
DEFAULT_CHUNK_SIZE_TOKENS = PIPELINE_MODEL_SETTINGS.pdf_chunk_tokenizer_model.chunk_size_tokens
DEFAULT_CHUNK_OVERLAP_TOKENS = PIPELINE_MODEL_SETTINGS.pdf_chunk_tokenizer_model.chunk_overlap_tokens
DEFAULT_MIN_CHUNK_SIZE_TOKENS = PIPELINE_MODEL_SETTINGS.pdf_chunk_tokenizer_model.min_chunk_size_tokens
DEFAULT_MAX_TABLE_TOKENS = PIPELINE_MODEL_SETTINGS.pdf_chunk_tokenizer_model.max_table_tokens
DEFAULT_TABLE_CAPTION_TOKENS = PIPELINE_MODEL_SETTINGS.pdf_chunk_tokenizer_model.table_caption_max_tokens


# ===== DATA CLASSES =====

@dataclass(frozen=True)
class PdfProcessorConfig:
    """Configuration for PDF extraction and chunking."""

    prompt: str = DEFAULT_EXTRACTION_PROMPT  # OpenAI system prompt
    # OCR configuration (OpenAI VLM)
    ocr_model: str = DEFAULT_OCR_MODEL
    ocr_dpi: int = DEFAULT_OCR_DPI
    ocr_max_image_dimension: int = DEFAULT_OCR_MAX_IMAGE_DIMENSION
    ocr_max_output_tokens: int = DEFAULT_OCR_MAX_OUTPUT_TOKENS
    ocr_concurrency: int = DEFAULT_OCR_CONCURRENCY
    ocr_temperature: float = DEFAULT_OCR_TEMPERATURE
    ocr_reasoning_effort: str = DEFAULT_OCR_REASONING_EFFORT

    # Chunking configuration
    token_model: str = DEFAULT_TOKEN_MODEL
    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    min_chunk_size_tokens: int = DEFAULT_MIN_CHUNK_SIZE_TOKENS
    max_table_tokens: Optional[int] = DEFAULT_MAX_TABLE_TOKENS
    table_caption_max_tokens: int = DEFAULT_TABLE_CAPTION_TOKENS


class DependencyError(RuntimeError):
    """Raised when a required optional dependency is missing."""


# ===== DEPENDENCY HELPERS =====
def ensure_dependencies(
    config: Optional[PdfProcessorConfig] = None,
    require_ocr_dependencies: bool = True,
) -> None:
    """Validate that required dependencies are available.

    NOTE: `require_ocr_dependencies` indicates whether PDF OCR dependencies
    (OPENAI_API_KEY) are required. It is True for PDF extraction,
    and False when using only the chunker on plain text.
    """
    missing = []

    if require_ocr_dependencies:
        if not os.environ.get("OPENAI_API_KEY"):
            missing.append("OPENAI_API_KEY environment variable is missing.")

    if missing:
        message = "Missing required dependencies:\n  - " + "\n  - ".join(missing)
        logger.error(message)
        raise DependencyError(message)


# ===== PDF EXTRACTION (OpenAI VLM) =====

def _pdf_to_images(pdf_path: Path, config: PdfProcessorConfig) -> List["Image.Image"]:
    """Render PDF pages to high-resolution images suitable for OCR."""
    logger.info("Rendering PDF '%s' at %d DPI", pdf_path.name, config.ocr_dpi)
    raw_images = convert_from_path(str(pdf_path), dpi=config.ocr_dpi)

    processed: List["Image.Image"] = []
    for idx, img in enumerate(raw_images):
        width, height = img.size
        longest_side = max(width, height)

        if longest_side > config.ocr_max_image_dimension:
            scale = config.ocr_max_image_dimension / float(longest_side)
            new_size = (int(width * scale), int(height * scale))
            img = img.resize(new_size, resample=Image.LANCZOS)
            logger.debug(
                "Page %d: downscaled from %dx%d to %s",
                idx + 1,
                width,
                height,
                new_size,
            )

        processed.append(img)

    logger.info("Rendered %d pages for OCR", len(processed))
    return processed


def _pil_image_to_base64_png(image: "Image.Image") -> str:
    """Encode a PIL image as a base64 PNG data URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{data}"


def _ocr_page(
    client: "OpenAI",
    image: "Image.Image",
    page_num: int,
    config: PdfProcessorConfig,
) -> Tuple[int, str]:
    """Worker function to OCR a single page with OpenAI VLM."""
    try:
        data_url = _pil_image_to_base64_png(image)

        logger.info(
            "Sending page %d to %s (reasoning=%s)...",
            page_num,
            config.ocr_model,
            config.ocr_reasoning_effort,
        )

        response = client.chat.completions.create(
            model=config.ocr_model,
            messages=[
                {"role": "system", "content": config.prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f'Transcribe page {page_num}. Start the output with: <!-- page: {page_num} -->',
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
            max_completion_tokens=config.ocr_max_output_tokens,
            temperature=config.ocr_temperature,
            reasoning_effort=config.ocr_reasoning_effort,
        )

        text = response.choices[0].message.content or ""
        logger.info("Page %d OCR complete: %d chars", page_num, len(text))
        return page_num, text.strip()
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.error("Error during OCR of page %d: %s", page_num, exc)
        return page_num, ""


def _transcribe_pdf_with_openai(pdf_path: Path, config: PdfProcessorConfig) -> List[str]:
    """Run OpenAI VLM OCR over all pages of a PDF in parallel."""
    client = OpenAI()
    images = _pdf_to_images(pdf_path, config)
    total_pages = len(images)

    results: Dict[int, str] = {}

    logger.info(
        "Starting parallel OCR for %d pages with %d workers",
        total_pages,
        config.ocr_concurrency,
    )

    with log_timing(logger, "openai_pdf_ocr"):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.ocr_concurrency
        ) as executor:
            future_to_page = {
                executor.submit(_ocr_page, client, img, idx + 1, config): idx
                for idx, img in enumerate(images)
            }

            for future in concurrent.futures.as_completed(future_to_page):
                page_num, text = future.result()
                results[page_num] = text

    # Reconstruct ordered list of page texts
    return [results[i] for i in range(1, total_pages + 1)]


def extract_pdf_to_markdown(pdf_path: Path, config: Optional[PdfProcessorConfig] = None) -> str:
    """Extract markdown text from a PDF using OpenAI VLM OCR."""
    cfg = config or PdfProcessorConfig()
    ensure_dependencies(cfg, require_ocr_dependencies=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at {pdf_path}")

    with log_timing(logger, "openai_pdf_ocr_total"):
        page_texts = _transcribe_pdf_with_openai(pdf_path, cfg)

    # Clean per-page outputs and join into a single markdown string.
    cleaned_pages = [t.strip() for t in page_texts if t and t.strip()]
    markdown_text = "\n\n".join(cleaned_pages)

    logger.info(
        "Extracted %s characters of markdown from %s via OpenAI VLM",
        f"{len(markdown_text):,}",
        pdf_path,
    )
    return markdown_text


# ===== TOKENIZATION HELPERS =====

def get_encoding(model_name: str) -> "tiktoken.Encoding":
    """Return a tiktoken encoding for the given model name."""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
        logger.debug("Using tiktoken encoding for model: %s", model_name)
        return encoding
    except KeyError:
        logger.warning("Unknown model '%s' for tiktoken. Falling back to cl100k_base.", model_name)
        return tiktoken.get_encoding("cl100k_base")


def build_length_function(encoding: "tiktoken.Encoding"):
    """Return a token-counting function compatible with LangChain splitters."""

    def length_fn(text: str) -> int:
        return len(encoding.encode(text))

    return length_fn


# ===== MARKDOWN SEGMENTATION =====

def segment_markdown_with_tables(text: str) -> List["Document"]:
    """
    Split markdown into logical blocks, preserving tables as atomic units.
    """
    lines = text.splitlines(keepends=True)
    blocks: List[Document] = []
    buffer: List[str] = []
    current_page: Optional[int] = None
    page_marker_re = re.compile(r"^<!--\s*page:\s*(\d+)\s*-->\s*$", re.IGNORECASE)

    def flush_buffer():
        if buffer:
            blocks.append(
                Document(
                    page_content="".join(buffer),
                    metadata={"block_type": "text", **({"page": current_page} if current_page else {})},
                )
            )
            buffer.clear()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.lstrip()

        # Page markers from OCR (e.g., "<!-- page: 3 -->") for later locator metadata.
        marker_match = page_marker_re.match(line.strip())
        if marker_match:
            flush_buffer()
            try:
                current_page = int(marker_match.group(1))
            except ValueError:
                current_page = None
            i += 1
            continue

        if (
            stripped.startswith("|")
            and i + 1 < n
            and re.match(r"^\s*\|?\s*:?-{3,}.*\|?.*$", lines[i + 1])
        ):
            flush_buffer()

            table_lines = [line, lines[i + 1]]
            i += 2

            while i < n and lines[i].lstrip().startswith("|"):
                table_lines.append(lines[i])
                i += 1

            blocks.append(
                Document(
                    page_content="".join(table_lines),
                    metadata={"block_type": "table", **({"page": current_page} if current_page else {})},
                )
            )
            continue

        buffer.append(line)
        i += 1

    flush_buffer()
    return blocks


def _extract_table_caption_text(prev_text: str, max_lines_back: int = 12) -> Optional[str]:
    """Try to extract a table caption-style snippet from the end of prev_text."""
    lines = prev_text.splitlines(keepends=True)
    if not lines:
        return None

    start_idx = max(0, len(lines) - max_lines_back)
    caption_start: Optional[int] = None

    for i in range(len(lines) - 1, start_idx - 1, -1):
        if re.match(r"^\s*Table\b", lines[i], flags=re.IGNORECASE):
            caption_start = i
            break

    if caption_start is None:
        return None

    return "".join(lines[caption_start:])


def _build_table_context_snippet(
    prev_text: str,
    encoding: "tiktoken.Encoding",
    max_tokens: int,
) -> Optional[str]:
    """Build a short context snippet to prepend to a table chunk."""
    if not prev_text or max_tokens <= 0:
        return None

    caption = _extract_table_caption_text(prev_text)
    candidate = caption if caption is not None else prev_text

    tokens = encoding.encode(candidate)
    if len(tokens) <= max_tokens:
        return candidate.strip()

    tail_tokens = tokens[-max_tokens:]
    return encoding.decode(tail_tokens).strip()


# ===== CHUNKER SETUP =====

def setup_chunker(length_function, chunk_size: int, chunk_overlap: int) -> "RecursiveCharacterTextSplitter":
    """Configure a token-aware RecursiveCharacterTextSplitter."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=length_function,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )


# ===== POST-PROCESSING =====

def merge_small_text_chunks(
    chunks: List["Document"],
    encoding: "tiktoken.Encoding",
    min_tokens: int,
    max_tokens: int,
) -> List["Document"]:
    """
    Merge neighbouring *text* chunks so we don't end up with tiny heading-only chunks.
    """
    if not chunks:
        return []

    merged: List[Document] = []
    current: Optional[Document] = None
    current_tokens: int = 0

    def token_len(doc: Document) -> int:
        return len(encoding.encode(doc.page_content))

    for ch in chunks:
        block_type = ch.metadata.get("block_type", "text")

        if block_type == "table":
            if current is not None:
                merged.append(current)
                current = None
                current_tokens = 0
            merged.append(ch)
            continue

        t_len = token_len(ch)

        if current is None:
            current = ch
            current_tokens = t_len
            continue

        current_page = current.metadata.get("page")
        next_page = ch.metadata.get("page")
        same_page = (
            (current_page == next_page)
            if (current_page is not None or next_page is not None)
            else True
        )

        if same_page and (current_tokens < min_tokens or t_len < min_tokens) and (
            current_tokens + t_len <= max_tokens + min_tokens
        ):
            current.page_content = (
                current.page_content.rstrip() + "\n\n" + ch.page_content.lstrip()
            )
            current_tokens += t_len
        else:
            merged.append(current)
            current = ch
            current_tokens = t_len

    if current is not None:
        merged.append(current)

    return merged

# ===== CHUNKING =====

def chunk_markdown_with_tables(
    text: str,
    chunker: "RecursiveCharacterTextSplitter",
    encoding: "tiktoken.Encoding",
    config: PdfProcessorConfig,
) -> List["Document"]:
    """Chunk markdown text while keeping tables intact and enriching them with context."""
    blocks = segment_markdown_with_tables(text)
    initial_chunks: List[Document] = []

    for idx, block in enumerate(blocks):
        block_type = block.metadata.get("block_type", "text")
        block.metadata.setdefault("block_index", idx)

        if block_type == "table":
            table_text = block.page_content
            context_snippet = None

            if config.table_caption_max_tokens > 0 and idx > 0:
                prev_block = blocks[idx - 1]
                if prev_block.metadata.get("block_type") == "text":
                    context_snippet = _build_table_context_snippet(
                        prev_text=prev_block.page_content,
                        encoding=encoding,
                        max_tokens=config.table_caption_max_tokens,
                    )

            enriched_text = (
                context_snippet.rstrip() + "\n\n" + table_text.lstrip()
                if context_snippet
                else table_text
            )

            base_meta = dict(block.metadata)
            base_meta["block_type"] = "table"
            base_meta["has_table_context"] = bool(context_snippet)

            if config.max_table_tokens is None:
                initial_chunks.append(
                    Document(
                        page_content=enriched_text,
                        metadata=base_meta,
                    )
                )
            else:
                token_len = len(encoding.encode(enriched_text))
                if token_len <= config.max_table_tokens:
                    initial_chunks.append(
                        Document(
                            page_content=enriched_text,
                            metadata=base_meta,
                        )
                    )
                else:
                    temp_doc = Document(
                        page_content=enriched_text,
                        metadata=base_meta,
                    )
                    table_chunks = chunker.split_documents([temp_doc])
                    for ch in table_chunks:
                        ch.metadata.setdefault("block_type", "table")
                        ch.metadata.setdefault("has_table_context", bool(context_snippet))
                    initial_chunks.extend(table_chunks)

        else:
            text_chunks = chunker.split_documents([block])
            for ch in text_chunks:
                ch.metadata.setdefault("block_type", "text")
            initial_chunks.extend(text_chunks)

    final_chunks = merge_small_text_chunks(
        initial_chunks,
        encoding=encoding,
        min_tokens=config.min_chunk_size_tokens,
        max_tokens=config.chunk_size_tokens,
    )
    return final_chunks


def chunk_text(text: str, config: Optional[PdfProcessorConfig] = None) -> List["Document"]:
    """High-level helper: token-based, table-aware chunking."""
    cfg = config or PdfProcessorConfig()
    ensure_dependencies(cfg, require_ocr_dependencies=False)

    if not text or not text.strip():
        return []

    encoding = get_encoding(cfg.token_model)
    chunker = setup_chunker(
        length_function=build_length_function(encoding),
        chunk_size=cfg.chunk_size_tokens,
        chunk_overlap=cfg.chunk_overlap_tokens,
    )

    with log_timing(logger, "chunk_markdown_with_tables"):
        chunks = chunk_markdown_with_tables(text, chunker, encoding, cfg)
    logger.info(
        "Chunked text into %d chunks (avg %.1f tokens)",
        len(chunks),
        (sum(len(encoding.encode(c.page_content)) for c in chunks) / len(chunks))
        if chunks
        else 0.0,
    )
    return chunks
