# Deep Research Newsletter Pipeline

This repository contains a local, agentic deep-research pipeline that turns a user-supplied paper into a structured newsletter draft. It ingests a PDF, extracts and chunks the article, reranks relevant chunks for question answering, supplements the paper with Tavily web research, and exports both markdown and structured JSON outputs.

This workflow supports [The Dose News](https://thedosenews.com/), a free AI-and-pharmacy newsletter for pharmacists. The live publication combines concise study breakdowns, an AI-generated audio briefing, and a NotebookLM workspace for follow-up questions and exploration. The exact release cadence is still evolving. Readers can subscribe by filling out the form on the website.

This codebase was originally bootstrapped with help from [ThinkDepthAI's Deep_Research](https://github.com/thinkdepthai/Deep_Research), then significantly adapted for this repository's paper-to-newsletter use case.

## What It Does

- Accepts a paper PDF as the primary input.
- Extracts article text with OpenAI-based PDF OCR.
- Chunks article content and reranks chunks with `BAAI/bge-reranker-v2-m3`.
- Runs a multi-step deep-research workflow over the paper and external web sources.
- Produces a final newsletter plus a structured JSON artifact in `outputs/`.

## Repository Structure

```text
src/             Core package
run_research_newsletter.py Entry script
papers/                    Place input PDFs here (kept empty in git)
outputs/                   Generated markdown and sample output examples
requirements.txt           Pip-installable runtime dependencies
pyproject.toml             Package metadata
```

## Requirements

- Python 3.11+
- An OpenAI API key
- A Tavily API key
- Poppler installed for `pdf2image`

### Poppler

`pdf2image` depends on Poppler utilities being available on the host system.

- macOS: `brew install poppler`
- Ubuntu/Debian: `sudo apt-get install poppler-utils`

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Environment Setup

Create a local `.env` file from the example template:

```bash
cp .env.example .env
```

Set:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Only those two environment variables are required for the default pipeline.

Optional observability:

- LangSmith is not required.
- If you already use LangSmith in your environment, you can enable tracing separately without changing the default pipeline setup.

## Inputs

Place the source paper PDF in `papers/`, or pass an absolute/relative path directly to the runner.

Examples:

```bash
python run_research_newsletter.py papers/my_paper.pdf
python run_research_newsletter.py papers/my_paper.pdf "Build a pharmacy-leader newsletter from the provided article."
```

If you pass only a PDF path, the script uses a default newsletter-building prompt.

## Outputs

Successful runs write artifacts to `outputs/`:

- `<paper-name>.md`: final markdown newsletter
- `<paper-name>.json`: structured newsletter export

The pipeline also writes runtime logs to `logs/` and caches OCR/chunking artifacts in `.deep_research_cache/`.

The public repo intentionally includes two sample markdown outputs to show the expected final artifact format:

- [FDA CDS newsletter example](outputs/cds.md), based on the FDA CDS guidance source article: [Clinical Decision Support Software](https://www.fda.gov/media/109618/download)
- [Incident-analysis newsletter example](outputs/incident.md), based on the source article: [Artificial intelligence-based incident analysis and learning system to enhance patient safety and improve treatment quality](https://doi.org/10.1038/s41746-026-02390-2)

Other generated outputs remain local-only.

## Reranker

The repository uses the `FlagEmbedding` cross-encoder reranker with model:

- `BAAI/bge-reranker-v2-m3`

Behavior and setup notes:

- First use downloads the reranker weights from Hugging Face.
- Device selection is automatic:
  - `mps` on Apple Silicon when available
  - `cuda` on NVIDIA GPUs when available
  - `cpu` otherwise
- CPU-only execution is supported, but reranking will be slower.

## Running The Pipeline

```bash
python run_research_newsletter.py papers/my_paper.pdf
```

## Optional LangSmith Trace Export

LangSmith is optional and is not required for the default pipeline.

If you already have LangSmith configured in your environment, the root utility `export_langsmith_trace.py` can export a full trace to JSON by trace ID.

Typical usage:

```bash
python export_langsmith_trace.py --trace-id <trace-uuid> --output full_trace.json
```

For optional LangSmith usage, set the usual LangSmith credentials in your local environment, for example `LANGSMITH_API_KEY`. This is separate from the default `.env.example`, which intentionally only documents `OPENAI_API_KEY` and `TAVILY_API_KEY`.

## Attribution

- Original repository inspiration: [thinkdepthai/Deep_Research](https://github.com/thinkdepthai/Deep_Research)
- Related background post: [Self-Balancing Agentic AI: Test-Time Diffusion Deep Research](https://paichunlin.substack.com/p/self-balancing-agentic-ai-test-time)
- Original author of those resources: Paichun Lin

This repository has been substantially modified from that starting point to support a different workflow, structure, and output format.

## License

This repository includes a root [LICENSE](LICENSE) file and should be distributed with that license text.
