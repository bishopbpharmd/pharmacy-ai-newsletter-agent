"""
Centralized Logging Configuration for Deep Research

This module provides consistent logging setup across all components of the
research system. All loggers write to a single log file for easy debugging
and correlation of events across components.

Usage:
    from src.logging_config import get_logger
    logger = get_logger("deep_research.my_module")
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Optional, Any, Callable

# ===== CONFIGURATION =====

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "research_agent.log"

# Log levels - can be controlled via environment variable
import os
LOG_LEVEL = os.environ.get("DEEP_RESEARCH_LOG_LEVEL", "INFO").upper()
VERBOSE_MODE = os.environ.get("DEEP_RESEARCH_VERBOSE", "false").lower() == "true"

# ===== SETUP =====

_initialized = False
_loggers = {}


def _ensure_log_dir():
    """Ensure log directory exists."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_file_handler() -> logging.FileHandler:
    """Create and configure the file handler."""
    _ensure_log_dir()
    handler = logging.FileHandler(LOG_FILE)
    handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    return handler


def _get_console_handler() -> logging.StreamHandler:
    """Create and configure the console handler."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        fmt="[%(levelname)s] %(name)s: %(message)s"
    ))
    return handler


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with consistent configuration.
    
    Args:
        name: Logger name (e.g., 'deep_research.supervisor')
        
    Returns:
        Configured logger instance
    """
    global _initialized, _loggers
    
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    
    # Only add handlers if this logger doesn't have any
    if not logger.handlers:
        logger.addHandler(_get_file_handler())
        logger.addHandler(_get_console_handler())
        logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        logger.propagate = False  # Prevent duplicate logging
    
    _loggers[name] = logger
    return logger


# ===== TIMING UTILITIES =====

@contextmanager
def log_timing(logger: logging.Logger, operation: str, level: int = logging.INFO):
    """
    Context manager to log timing of operations.
    
    Usage:
        with log_timing(logger, "LLM call"):
            result = model.invoke(messages)
    
    Args:
        logger: Logger instance to use
        operation: Description of the operation being timed
        level: Log level for the timing message
    """
    start = time.perf_counter()
    logger.log(level, "%s STARTED", operation)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.log(level, "%s COMPLETED in %.2fs", operation, elapsed)


def timed(logger: logging.Logger, operation: Optional[str] = None):
    """
    Decorator to log timing of function calls.
    
    Usage:
        @timed(logger, "process_results")
        def process_results(data):
            ...
    
    Args:
        logger: Logger instance to use
        operation: Optional operation name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start = time.perf_counter()
            logger.debug("%s STARTED", op_name)
            try:
                result = func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug("%s COMPLETED in %.2fs", op_name, elapsed)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error("%s FAILED after %.2fs: %s", op_name, elapsed, e)
                raise
        return wrapper
    return decorator


async def timed_async(logger: logging.Logger, operation: Optional[str] = None):
    """
    Decorator for async functions timing.
    
    Args:
        logger: Logger instance to use  
        operation: Optional operation name (defaults to function name)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            op_name = operation or func.__name__
            start = time.perf_counter()
            logger.debug("%s STARTED", op_name)
            try:
                result = await func(*args, **kwargs)
                elapsed = time.perf_counter() - start
                logger.debug("%s COMPLETED in %.2fs", op_name, elapsed)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                logger.error("%s FAILED after %.2fs: %s", op_name, elapsed, e)
                raise
        return wrapper
    return decorator


# ===== MESSAGE UTILITIES =====

def summarize_message(msg: Any, max_content_len: int = 200) -> str:
    """
    Create a compact summary of a LangChain message for logging.
    
    Args:
        msg: Message object (HumanMessage, AIMessage, ToolMessage, etc.)
        max_content_len: Maximum length of content preview
        
    Returns:
        Compact string summary of the message
    """
    msg_type = type(msg).__name__
    content = getattr(msg, 'content', '')
    content_str = str(content)
    content_preview = (content_str[:max_content_len] + '...') if len(content_str) > max_content_len else content_str
    tool_calls = getattr(msg, 'tool_calls', None)
    name = getattr(msg, 'name', None)
    
    parts = [msg_type]
    if name:
        parts.append(f"[{name}]")
    if tool_calls:
        tc_names = [tc.get('name', '?') for tc in tool_calls]
        parts.append(f"tool_calls={tc_names}")
    parts.append(f"content_len={len(content_str)}")
    
    return " ".join(parts)


def log_messages(logger: logging.Logger, messages: list, prefix: str = "msg", level: int = logging.DEBUG):
    """
    Log a summary of a list of messages.
    
    Args:
        logger: Logger instance
        messages: List of message objects
        prefix: Prefix for log lines
        level: Log level
    """
    for i, msg in enumerate(messages):
        logger.log(level, "  %s[%d]: %s", prefix, i, summarize_message(msg))


# ===== STATISTICS UTILITIES =====

def log_research_stats(logger: logging.Logger, 
                       total_searches: int = 0,
                       total_results_chars: int = 0,
                       llm_calls: int = 0,
                       errors: int = 0,
                       elapsed_time: float = 0.0):
    """
    Log summary statistics for a research session.
    
    Args:
        logger: Logger instance
        total_searches: Number of web searches performed
        total_results_chars: Total characters of search results
        llm_calls: Number of LLM API calls
        errors: Number of errors encountered
        elapsed_time: Total elapsed time in seconds
    """
    logger.info("="*60)
    logger.info("RESEARCH STATISTICS")
    logger.info("  Total searches:     %d", total_searches)
    logger.info("  Results volume:     %d chars (%.1f KB)", total_results_chars, total_results_chars/1024)
    logger.info("  LLM calls:          %d", llm_calls)
    logger.info("  Errors:             %d", errors)
    logger.info("  Total time:         %.1fs (%.1f min)", elapsed_time, elapsed_time/60)
    logger.info("="*60)


# ===== SEPARATOR UTILITIES =====

def log_section_start(logger: logging.Logger, section: str, char: str = "=", width: int = 80):
    """Log a section start marker for visual clarity."""
    logger.info(char * width)
    logger.info("%s START", section)


def log_section_end(logger: logging.Logger, section: str, char: str = "-", width: int = 60):
    """Log a section end marker."""
    logger.info("%s END", section)
    logger.info(char * width)


# ===== TOKEN USAGE TRACKING =====

class TokenUsageTracker:
    """
    Tracks cumulative token usage across multiple LLM calls.
    
    Usage:
        tracker = TokenUsageTracker()
        # After each LLM call:
        tracker.add_usage(response, "summarization")
        # At the end:
        tracker.log_summary(logger)
    """
    
    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0
        self.by_operation = {}  # operation_name -> {prompt, completion, total, count}
    
    def add_usage(self, response: Any, operation: str = "llm_call") -> dict:
        """
        Extract and accumulate token usage from a LangChain response.
        
        Args:
            response: LangChain AIMessage or similar response object
            operation: Name of the operation for breakdown tracking
            
        Returns:
            Dict with usage for this call, or empty dict if not available
        """
        usage = extract_token_usage(response)
        if not usage:
            return {}
        
        prompt = usage.get("prompt_tokens", 0)
        completion = usage.get("completion_tokens", 0)
        total = usage.get("total_tokens", 0)
        
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += total
        self.call_count += 1
        
        # Track by operation
        if operation not in self.by_operation:
            self.by_operation[operation] = {"prompt": 0, "completion": 0, "total": 0, "count": 0}
        self.by_operation[operation]["prompt"] += prompt
        self.by_operation[operation]["completion"] += completion
        self.by_operation[operation]["total"] += total
        self.by_operation[operation]["count"] += 1
        
        return usage
    
    def log_summary(self, logger: logging.Logger):
        """Log a summary of all token usage."""
        logger.info("="*60)
        logger.info("TOKEN USAGE SUMMARY")
        logger.info("  Total calls:            %d", self.call_count)
        logger.info("  Total prompt tokens:    %d", self.total_prompt_tokens)
        logger.info("  Total completion tokens: %d", self.total_completion_tokens)
        logger.info("  Total tokens:           %d", self.total_tokens)
        
        if self.by_operation:
            logger.info("  --- By Operation ---")
            for op, stats in sorted(self.by_operation.items()):
                logger.info("    %s: %d calls, %d tokens (prompt=%d, completion=%d)",
                           op, stats["count"], stats["total"], stats["prompt"], stats["completion"])
        logger.info("="*60)
    
    def get_summary(self) -> dict:
        """Get summary as a dictionary."""
        return {
            "total_calls": self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "by_operation": self.by_operation.copy()
        }


def extract_token_usage(response: Any) -> dict:
    """
    Extract token usage from a LangChain response.
    
    Args:
        response: LangChain AIMessage or similar response object
        
    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens or empty dict
    """
    if response is None:
        return {}
    
    # Try response_metadata (standard LangChain location)
    metadata = getattr(response, 'response_metadata', {}) or {}
    
    # OpenAI format
    usage = metadata.get('token_usage') or metadata.get('usage') or {}
    if usage:
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "model": metadata.get("model_name", metadata.get("model", "unknown"))
        }
    
    # Try usage_metadata (newer LangChain)
    usage_meta = getattr(response, 'usage_metadata', None)
    if usage_meta:
        return {
            "prompt_tokens": getattr(usage_meta, 'input_tokens', 0),
            "completion_tokens": getattr(usage_meta, 'output_tokens', 0),
            "total_tokens": getattr(usage_meta, 'total_tokens', 0),
            "model": "unknown"
        }
    
    return {}


def log_token_usage(logger: logging.Logger, response: Any, operation: str = "LLM call"):
    """
    Log token usage from a single LLM response.
    
    Args:
        logger: Logger instance
        response: LangChain response object
        operation: Description of the operation
    """
    usage = extract_token_usage(response)
    if usage:
        logger.info("  TOKEN_USAGE [%s]: prompt=%d, completion=%d, total=%d, model=%s",
                   operation,
                   usage.get("prompt_tokens", 0),
                   usage.get("completion_tokens", 0),
                   usage.get("total_tokens", 0),
                   usage.get("model", "unknown"))
    else:
        logger.debug("  TOKEN_USAGE [%s]: not available", operation)


# Global tracker for session-wide usage
_global_tracker = TokenUsageTracker()


def get_global_tracker() -> TokenUsageTracker:
    """Get the global token usage tracker."""
    return _global_tracker


def reset_global_tracker():
    """Reset the global token usage tracker."""
    global _global_tracker
    _global_tracker = TokenUsageTracker()
