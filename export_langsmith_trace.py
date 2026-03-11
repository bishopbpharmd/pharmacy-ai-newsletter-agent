#!/usr/bin/env python3
"""Export a full LangSmith trace to a single JSON file."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Export all LangSmith runs for a trace into one JSON file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trace-id",
        required=True,
        type=UUID,
        help=(
            "Trace UUID to export. If a child run UUID is provided instead, "
            "the script resolves its parent trace first."
        ),
    )
    parser.add_argument(
        "--output",
        default="full_trace.json",
        help="Output path for the exported JSON file.",
    )
    return parser.parse_args(argv)


def now_utc_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def load_local_env() -> None:
    """Load a local .env file when present so the CLI works outside exported shells."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent / ".env",
    ]

    for env_path in candidates:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
            return

    load_dotenv(override=False)


def has_langsmith_api_key() -> bool:
    """Return whether a LangSmith API key is available in the environment."""
    return bool(os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY"))


def model_to_dict(value: Any) -> Any:
    """Convert Pydantic models to dictionaries across SDK/Pydantic versions."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    return value


def make_json_safe(value: Any) -> Any:
    """Recursively convert SDK values into JSON-safe primitives."""
    value = model_to_dict(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.isoformat()
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, Decimal):
        return str(value)

    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "encoding": "base64",
            "data": base64.b64encode(value).decode("ascii"),
        }

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(make_json_safe(key)): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    return repr(value)


def sort_key(run: Any) -> tuple[str, str, str]:
    """Sort runs in trace execution order using dotted_order first."""
    start_time = getattr(run, "start_time", None)
    return (
        getattr(run, "dotted_order", "") or "",
        start_time.isoformat() if start_time else "",
        str(getattr(run, "id", "")),
    )


def immediate_parent_id(run: Any) -> UUID | None:
    """Return the nearest parent run ID when present."""
    parent_run_id = getattr(run, "parent_run_id", None)
    if parent_run_id is not None:
        return parent_run_id

    parent_run_ids = getattr(run, "parent_run_ids", None) or []
    return parent_run_ids[-1] if parent_run_ids else None


def collect_runs(run_iterator: Any, trace_id: UUID) -> tuple[list[Any], str | None]:
    """Collect runs from an iterator and preserve partial results on late failures."""
    runs: list[Any] = []

    try:
        for run in run_iterator:
            runs.append(run)
    except Exception as exc:
        if not runs:
            raise RuntimeError(f"Failed to list runs for trace {trace_id}: {exc}") from exc
        return (
            runs,
            f"LangSmith returned an error after {len(runs)} runs; exporting the partial "
            f"trace collected so far. Original error: {exc}",
        )

    return runs, None


def fetch_trace_runs(client: Any, provided_id: UUID) -> tuple[UUID, list[Any], str | None]:
    """Fetch all runs for a trace, resolving child run IDs to trace IDs when needed."""
    trace_id = provided_id

    runs, warning = collect_runs(client.list_runs(trace_id=trace_id), trace_id)

    if runs:
        return trace_id, runs, warning

    # If the supplied UUID is a child run ID, resolve its owning trace and re-list.
    try:
        resolved_run = client.read_run(provided_id)
    except Exception as exc:
        raise RuntimeError(
            f"No runs found for {provided_id}, and resolving it as a run ID failed: {exc}"
        ) from exc

    trace_id = getattr(resolved_run, "trace_id", None)
    if trace_id is None:
        raise RuntimeError(
            f"No runs found for {provided_id}, and the resolved run does not include a trace_id."
        )

    runs, warning = collect_runs(client.list_runs(trace_id=trace_id), trace_id)
    return trace_id, runs, warning


def build_direct_children(runs: list[Any]) -> dict[UUID, list[UUID]]:
    """Build immediate child relationships from the flat run list."""
    direct_children: dict[UUID, list[UUID]] = defaultdict(list)

    for run in runs:
        direct_children[getattr(run, "id")]

    for run in runs:
        parent_id = immediate_parent_id(run)
        if parent_id is not None:
            direct_children[parent_id].append(getattr(run, "id"))

    return dict(direct_children)


def serialize_run(run: Any, direct_children: dict[UUID, list[UUID]]) -> dict[str, Any]:
    """Keep the exported run payload explicit and critique-friendly."""
    parent_run_id = getattr(run, "parent_run_id", None)
    parent_run_ids = getattr(run, "parent_run_ids", None) or (
        [parent_run_id] if parent_run_id else []
    )
    immediate_children = direct_children.get(getattr(run, "id"), [])
    child_run_ids = getattr(run, "child_run_ids", None) or immediate_children

    run_payload = {
        "id": getattr(run, "id", None),
        "trace_id": getattr(run, "trace_id", None),
        "name": getattr(run, "name", None),
        "run_type": getattr(run, "run_type", None),
        "inputs": getattr(run, "inputs", None),
        "outputs": getattr(run, "outputs", None),
        "error": getattr(run, "error", None),
        "start_time": getattr(run, "start_time", None),
        "end_time": getattr(run, "end_time", None),
        "dotted_order": getattr(run, "dotted_order", None),
        "parent_run_id": parent_run_id,
        "parent_run_ids": parent_run_ids,
        "direct_child_run_ids": immediate_children,
        "child_run_ids": child_run_ids,
        "tags": getattr(run, "tags", None),
        "extra": getattr(run, "extra", None),
        "events": getattr(run, "events", None),
        "status": getattr(run, "status", None),
        "serialized": getattr(run, "serialized", None),
        "latency_seconds": getattr(run, "latency", None),
        "prompt_tokens": getattr(run, "prompt_tokens", None),
        "completion_tokens": getattr(run, "completion_tokens", None),
        "total_tokens": getattr(run, "total_tokens", None),
        "prompt_cost": getattr(run, "prompt_cost", None),
        "completion_cost": getattr(run, "completion_cost", None),
        "total_cost": getattr(run, "total_cost", None),
        "first_token_time": getattr(run, "first_token_time", None),
        "feedback_stats": getattr(run, "feedback_stats", None),
        "app_path": getattr(run, "app_path", None),
        "url": getattr(run, "url", None),
    }
    return make_json_safe(run_payload)


def export_trace(trace_id: UUID, output_path: Path) -> int:
    """Export a trace to the requested JSON path."""
    load_local_env()

    try:
        from langsmith import Client
    except ImportError:
        print(
            "Error: langsmith is not installed in this environment. Install it with "
            "`pip install langsmith`.",
            file=sys.stderr,
        )
        return 1

    if not has_langsmith_api_key():
        print(
            "Error: LANGSMITH_API_KEY is not set at runtime. This script now attempts "
            "to load a local .env file automatically, but no LangSmith API key was "
            "found in the process environment or .env.",
            file=sys.stderr,
        )
        return 1

    client = Client()

    try:
        resolved_trace_id, runs, warning = fetch_trace_runs(client, trace_id)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if "Authentication failed" in str(exc):
            print(
                "Hint: LangSmith rejected the API key at runtime. "
                "LANGSMITH_PROJECT does not affect authentication, and "
                "LANGSMITH_WORKSPACE_ID is only needed for org-scoped keys "
                "(that case usually returns a 403 workspace error, not a 401 invalid token).",
                file=sys.stderr,
            )
        return 1

    if not runs:
        print(f"Error: no runs found for trace {trace_id}.", file=sys.stderr)
        return 1

    ordered_runs = sorted(runs, key=sort_key)
    direct_children = build_direct_children(ordered_runs)

    payload = {
        "trace_id": str(resolved_trace_id),
        "exported_at": now_utc_iso(),
        "runs": [serialize_run(run, direct_children) for run in ordered_runs],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    if warning:
        print(f"Warning: {warning}", file=sys.stderr)

    print(f"Exported {len(ordered_runs)} runs from trace {resolved_trace_id} to {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    return export_trace(trace_id=args.trace_id, output_path=Path(args.output))


if __name__ == "__main__":
    raise SystemExit(main())
