"""Central model and runtime configuration for the newsletter pipeline."""

from __future__ import annotations

from dataclasses import dataclass, fields
import importlib.util
import os
import sys
from typing import Optional


class PipelineConfigurationError(RuntimeError):
    """Raised when required model configuration or runtime dependencies are missing."""


@dataclass(frozen=True)
class ChatModelSetting:
    """Configuration for a LangChain chat model."""

    purpose: str
    model_name: str
    max_tokens: Optional[int] = None
    timeout_seconds: Optional[float] = None
    max_retries: Optional[int] = None


@dataclass(frozen=True)
class PdfOcrModelSetting:
    """Configuration for PDF OCR via the OpenAI API."""

    purpose: str
    model_name: str
    dpi: int = 300
    max_image_dimension: int = 4096
    max_output_tokens: int = 16384
    concurrency: int = 4
    temperature: float = 1.0
    reasoning_effort: str = "low"


@dataclass(frozen=True)
class PdfChunkingTokenizerSetting:
    """Configuration for token-aware PDF chunking."""

    purpose: str
    model_name: str
    chunk_size_tokens: int = 384
    chunk_overlap_tokens: int = 64
    min_chunk_size_tokens: int = 64
    max_table_tokens: Optional[int] = None
    table_caption_max_tokens: int = 250


@dataclass(frozen=True)
class RerankerModelSetting:
    """Configuration for the document chunk reranker."""

    purpose: str
    model_name: str


@dataclass(frozen=True)
class PipelineModelSettings:
    """Single-source model configuration for the active newsletter pipeline."""

    scope_template_selection_and_brief_model: ChatModelSetting = ChatModelSetting(
        purpose="Template selection, clarification handling, and research brief generation.",
        model_name="openai:gpt-5-mini",
    )
    scope_draft_generation_model: ChatModelSetting = ChatModelSetting(
        purpose="Initial article-first draft generation before deeper research.",
        model_name="openai:gpt-5-mini",
    )
    research_agent_tool_call_model: ChatModelSetting = ChatModelSetting(
        purpose="Legacy research-agent tool-calling loop in src/research_agent.py.",
        model_name="openai:gpt-5-mini",
    )
    research_agent_summarization_model: ChatModelSetting = ChatModelSetting(
        purpose="Legacy research-agent answer summarization in src/research_agent.py.",
        model_name="openai:gpt-5-mini",
    )
    research_agent_compression_model: ChatModelSetting = ChatModelSetting(
        purpose="Legacy research-agent final compression pass in src/research_agent.py.",
        model_name="openai:gpt-5-mini",
        max_tokens=32000,
    )
    storm_perspective_discovery_model: ChatModelSetting = ChatModelSetting(
        purpose="STORM perspective discovery.",
        model_name="openai:gpt-5-mini",
    )
    storm_research_plan_model: ChatModelSetting = ChatModelSetting(
        purpose="STORM perspective-specific research plan generation.",
        model_name="openai:gpt-5-mini",
    )
    storm_writer_model: ChatModelSetting = ChatModelSetting(
        purpose="STORM writer questions and tool-calling turns.",
        model_name="openai:gpt-5-mini",
    )
    storm_expert_synthesis_model: ChatModelSetting = ChatModelSetting(
        purpose="STORM expert synthesis over retrieved evidence.",
        model_name="openai:gpt-5-mini",
    )
    storm_retry_reflection_model: ChatModelSetting = ChatModelSetting(
        purpose="STORM retry and answer-quality reflection.",
        model_name="openai:gpt-5-mini",
    )
    storm_answer_synthesis_model: ChatModelSetting = ChatModelSetting(
        purpose="STORM answer synthesis across retries.",
        model_name="openai:gpt-5-mini",
    )
    storm_compression_model: ChatModelSetting = ChatModelSetting(
        purpose="STORM compressed research summary generation.",
        model_name="openai:gpt-5-mini",
        max_tokens=32000,
    )
    supervisor_orchestration_model: ChatModelSetting = ChatModelSetting(
        purpose="Supervisor orchestration, delegation, and stop/continue decisions.",
        model_name="openai:gpt-5-mini",
        timeout_seconds=70,
        max_retries=1,
    )
    final_report_generation_model: ChatModelSetting = ChatModelSetting(
        purpose="Final newsletter synthesis.",
        model_name="openai:gpt-5.4",
        max_tokens=40000,
    )
    final_copywriter_polish_model: ChatModelSetting = ChatModelSetting(
        purpose="Final copyediting and polish pass.",
        model_name="openai:gpt-5-mini",
        max_tokens=8000,
    )
    final_critique_reflection_model: ChatModelSetting = ChatModelSetting(
        purpose="Critique-and-rewrite quality gate for the final draft.",
        model_name="openai:gpt-5.4",
    )
    webpage_summarization_model: ChatModelSetting = ChatModelSetting(
        purpose="Summarizing Tavily webpage content.",
        model_name="openai:gpt-5-mini",
    )
    draft_refinement_model: ChatModelSetting = ChatModelSetting(
        purpose="Draft refinement after research findings are incorporated.",
        model_name="openai:gpt-5-mini",
        max_tokens=32000,
    )
    article_ingest_summary_model: ChatModelSetting = ChatModelSetting(
        purpose="Global article summary generation at ingest time.",
        model_name="openai:gpt-5.4",
        max_tokens=5000,
    )
    pdf_ocr_transcription_model: PdfOcrModelSetting = PdfOcrModelSetting(
        purpose="PDF page transcription via OpenAI vision/chat OCR.",
        model_name="gpt-5.4",
    )
    pdf_chunk_tokenizer_model: PdfChunkingTokenizerSetting = PdfChunkingTokenizerSetting(
        purpose="Tokenizer selection for PDF chunk sizing and overlap calculations.",
        model_name="gpt-5.4",
    )
    document_chunk_reranker_model: RerankerModelSetting = RerankerModelSetting(
        purpose="Cross-encoder reranker for article chunk retrieval.",
        model_name="BAAI/bge-reranker-v2-m3",
    )


PIPELINE_MODEL_SETTINGS = PipelineModelSettings()


def build_chat_model(setting: ChatModelSetting):
    """Instantiate a LangChain chat model from centralized configuration."""
    from langchain.chat_models import init_chat_model

    kwargs = {"model": setting.model_name}
    if setting.max_tokens is not None:
        kwargs["max_tokens"] = setting.max_tokens
    if setting.timeout_seconds is not None:
        kwargs["timeout"] = setting.timeout_seconds
    if setting.max_retries is not None:
        kwargs["max_retries"] = setting.max_retries
    return init_chat_model(**kwargs)


def _module_available(module_name: str) -> bool:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        spec = None
    return spec is not None or module_name in sys.modules


def validate_pipeline_configuration(
    config: PipelineModelSettings = PIPELINE_MODEL_SETTINGS,
    require_environment_variables: bool = True,
    require_python_dependencies: bool = True,
) -> None:
    """Hard-fail on missing model settings or required runtime prerequisites."""

    errors: list[str] = []

    for field in fields(config):
        setting = getattr(config, field.name)

        if isinstance(setting, ChatModelSetting):
            if not str(setting.model_name).strip():
                errors.append(
                    f"{field.name}: model_name is blank. This setting controls {setting.purpose}"
                )
            if setting.max_tokens is not None and setting.max_tokens <= 0:
                errors.append(
                    f"{field.name}: max_tokens must be > 0 when provided."
                )
            if setting.timeout_seconds is not None and setting.timeout_seconds <= 0:
                errors.append(f"{field.name}: timeout_seconds must be > 0 when provided.")
            if setting.max_retries is not None and setting.max_retries < 0:
                errors.append(f"{field.name}: max_retries must be >= 0 when provided.")
            continue

        if isinstance(setting, PdfOcrModelSetting):
            if not str(setting.model_name).strip():
                errors.append(
                    f"{field.name}: model_name is blank. This setting controls {setting.purpose}"
                )
            if setting.dpi <= 0:
                errors.append(f"{field.name}: dpi must be > 0.")
            if setting.max_image_dimension <= 0:
                errors.append(f"{field.name}: max_image_dimension must be > 0.")
            if setting.max_output_tokens <= 0:
                errors.append(f"{field.name}: max_output_tokens must be > 0.")
            if setting.concurrency <= 0:
                errors.append(f"{field.name}: concurrency must be > 0.")
            if not str(setting.reasoning_effort).strip():
                errors.append(f"{field.name}: reasoning_effort must be non-empty.")
            continue

        if isinstance(setting, PdfChunkingTokenizerSetting):
            if not str(setting.model_name).strip():
                errors.append(
                    f"{field.name}: model_name is blank. This setting controls {setting.purpose}"
                )
            if setting.chunk_size_tokens <= 0:
                errors.append(f"{field.name}: chunk_size_tokens must be > 0.")
            if setting.chunk_overlap_tokens < 0:
                errors.append(f"{field.name}: chunk_overlap_tokens must be >= 0.")
            if setting.min_chunk_size_tokens <= 0:
                errors.append(f"{field.name}: min_chunk_size_tokens must be > 0.")
            if setting.table_caption_max_tokens <= 0:
                errors.append(f"{field.name}: table_caption_max_tokens must be > 0.")
            continue

        if isinstance(setting, RerankerModelSetting):
            if not str(setting.model_name).strip():
                errors.append(
                    f"{field.name}: model_name is blank. This setting controls {setting.purpose}"
                )

    if require_environment_variables:
        if not os.getenv("OPENAI_API_KEY"):
            errors.append(
                "OPENAI_API_KEY is required for the configured OpenAI chat and OCR models."
            )
        if not os.getenv("TAVILY_API_KEY"):
            errors.append(
                "TAVILY_API_KEY is required for Tavily-backed external research steps."
            )

    if require_python_dependencies:
        required_modules = {
            "langchain.chat_models": "LangChain chat model factory",
            "openai": "OpenAI Python client",
            "tavily": "Tavily Python client",
            "FlagEmbedding": "FlagEmbedding reranker package",
            "torch": "Torch runtime required by FlagEmbedding",
            "pdf2image": "pdf2image PDF rasterization package",
            "PIL": "Pillow image package",
            "tiktoken": "tiktoken tokenizer package",
            "langchain_text_splitters": "LangChain text splitter package",
        }
        for module_name, description in required_modules.items():
            if not _module_available(module_name):
                errors.append(
                    f"Missing Python dependency for {description}: import '{module_name}' could not be resolved."
                )

    if errors:
        joined = "\n".join(f"  - {error}" for error in errors)
        raise PipelineConfigurationError(
            "Newsletter pipeline configuration preflight failed.\n"
            f"Review src/model_config.py and the runtime environment:\n{joined}"
        )
