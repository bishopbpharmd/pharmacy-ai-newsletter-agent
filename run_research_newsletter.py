#!/usr/bin/env python3
"""
Interactive Newsletter Builder Script

Run the pharmacy-leader newsletter workflow and interact with the agent when clarification is needed.
Uses LangGraph's interrupt mechanism for human-in-the-loop feedback.
"""

from dotenv import load_dotenv
import os
import sys
import asyncio
import warnings
import time
from pathlib import Path

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Import required modules
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from src.model_config import validate_pipeline_configuration
from src.logging_config import get_logger, get_global_tracker, reset_global_tracker
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# ===== LOGGING SETUP =====

logger = get_logger("deep_research.runner")

console = Console()


def display_final_report(report: str):
    """Display the final newsletter"""
    console.print("\n[bold green]═══════════════════════════════════════════════════════════════[/bold green]")
    console.print("[bold green]                      FINAL REPORT                              [/bold green]")
    console.print("[bold green]═══════════════════════════════════════════════════════════════[/bold green]\n")
    console.print(Panel(Markdown(report), title="Newsletter Draft", border_style="green"))


def display_clarification(question: str):
    """Display a clarification question from the agent"""
    console.print("\n[bold yellow]═══════════════════════════════════════════════════════════════[/bold yellow]")
    console.print("[bold yellow]                  CLARIFICATION NEEDED                          [/bold yellow]")
    console.print("[bold yellow]═══════════════════════════════════════════════════════════════[/bold yellow]\n")
    console.print(Panel(Markdown(question), title="Agent Question", border_style="yellow"))


def get_user_response() -> str:
    """Get user's response to a clarification question"""
    console.print("\n[bold cyan]Your response (or 'quit' to exit):[/bold cyan]")
    response = input("> ").strip()
    return response


async def run_research(query: str, thread_id: str = "default", source_filename: str | None = None, pdf_path: str | None = None):
    """Run the research agent with interactive clarification handling"""
    validate_pipeline_configuration()
    from src.research_agent_full import deep_researcher_builder
    
    logger.info("="*80)
    logger.info("RUN_RESEARCH START")
    logger.info("  query: %s", query[:200] + "..." if len(query) > 200 else query)
    logger.info("  thread_id: %s", thread_id)
    logger.info("  source_filename: %s", source_filename)
    logger.info("  pdf_path: %s", pdf_path)
    
    total_start_time = time.perf_counter()
    
    # Initialize agent with checkpointer for state persistence
    checkpointer = MemorySaver()
    full_agent = deep_researcher_builder.compile(checkpointer=checkpointer)
    logger.info("  Agent compiled with checkpointer")
    
    # Config includes thread_id for state persistence and recursion_limit to handle
    # deep research loops (supervisor iterations + researcher subgraph loops)
    thread = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100  # Default is 25, which is too low for complex research
    }
    
    console.print(f"\n[bold blue]Starting newsletter build:[/bold blue] {query}")
    console.print("[dim]This run adapts the deep-research flow for concise pharmacy-leader newsletters.[/dim]\n")
    
    # Initial invocation - include pdf_path and source_filename if provided
    current_input = {"messages": [HumanMessage(content=query)]}
    if pdf_path:
        current_input["pdf_path"] = pdf_path
    if source_filename:
        current_input["source_filename"] = source_filename
    
    max_iterations = 10  # Prevent infinite loops
    iteration = 0
    
    while iteration < max_iterations:
        iteration += 1
        iteration_start = time.perf_counter()
        
        logger.info("-"*60)
        logger.info("RUNNER ITERATION %d/%d START", iteration, max_iterations)
        
        try:
            # Run the agent
            logger.info("  Invoking full_agent.ainvoke...")
            invoke_start = time.perf_counter()
            result = await full_agent.ainvoke(current_input, config=thread)
            invoke_elapsed = time.perf_counter() - invoke_start
            
            logger.info("  ainvoke completed in %.2fs", invoke_elapsed)
            logger.info("  result_keys: %s", list(result.keys()) if result else "None")
            
            # Check for final report
            if "final_report" in result and result.get("final_report"):
                report_len = len(result["final_report"])
                total_elapsed = time.perf_counter() - total_start_time
                
                logger.info("="*80)
                logger.info("FINAL REPORT RECEIVED")
                logger.info("  report_length: %d chars", report_len)
                logger.info("  total_time: %.2fs (%.1f minutes)", total_elapsed, total_elapsed/60)
                logger.info("  iterations: %d", iteration)
                logger.info("="*80)
                
                display_final_report(result["final_report"])
                return result
            
            # Check agent state for interrupts
            state = await full_agent.aget_state(thread)
            logger.debug("  Agent state retrieved, checking for interrupts...")
            
            # If there are pending tasks (interrupts), handle them
            if state.tasks:
                logger.info("  Found %d pending tasks", len(state.tasks))
                for task in state.tasks:
                    if hasattr(task, 'interrupts') and task.interrupts:
                        logger.info("  Task has %d interrupts", len(task.interrupts))
                        for interrupt_data in task.interrupts:
                            # Display the clarification question
                            if hasattr(interrupt_data, 'value') and isinstance(interrupt_data.value, dict):
                                question = interrupt_data.value.get("question", str(interrupt_data.value))
                            else:
                                question = str(interrupt_data)
                            
                            logger.info("  INTERRUPT - clarification needed: %s", question[:100])
                            display_clarification(question)
                            
                            # Get user response
                            user_response = get_user_response()
                            logger.info("  User response: %s", user_response[:100] if user_response else "None")
                            
                            if user_response.lower() in ['quit', 'exit', 'q']:
                                logger.info("  User requested exit")
                                console.print("[yellow]Exiting...[/yellow]")
                                return result
                            
                            # Resume with user's response using Command
                            console.print("\n[blue]Processing your response...[/blue]\n")
                            current_input = Command(resume=user_response)
                            break
                    else:
                        logger.debug("  Task has no interrupts")
                        break
            else:
                logger.info("  No pending tasks")
                # No tasks pending. Inspect result and supervisor messages for questions.
                # 1) Direct messages in result
                if "messages" in result and result["messages"]:
                    last_message = result["messages"][-1]
                    if hasattr(last_message, 'content') and last_message.content:
                        content = last_message.content
                        logger.debug("  Last message content: %s...", content[:100])
                        if "?" in content or "option" in content.lower() or "choose" in content.lower():
                            logger.info("  Detected question in messages, prompting user")
                            display_clarification(content)
                            user_response = get_user_response()
                            
                            if user_response.lower() in ['quit', 'exit', 'q']:
                                logger.info("  User requested exit")
                                console.print("[yellow]Exiting...[/yellow]")
                                return result
                            
                            console.print("\n[blue]Processing your response...[/blue]\n")
                            current_input = {"messages": [HumanMessage(content=user_response)]}
                            continue

                # 2) Supervisor questions (not surfaced as tasks)
                supervisor_messages = state.values.get("supervisor_messages") if hasattr(state, "values") else state.get("supervisor_messages")
                if supervisor_messages:
                    logger.debug("  Checking %d supervisor messages for questions", len(supervisor_messages))
                    last_sup = supervisor_messages[-1]
                    sup_content = getattr(last_sup, "content", "")
                    if sup_content and ("?" in sup_content or "option" in sup_content.lower() or "choose" in sup_content.lower()):
                        logger.info("  Detected question in supervisor messages")
                        display_clarification(sup_content)
                        user_response = get_user_response()

                        if user_response.lower() in ['quit', 'exit', 'q']:
                            logger.info("  User requested exit")
                            console.print("[yellow]Exiting...[/yellow]")
                            return result

                        console.print("\n[blue]Processing your response...[/blue]\n")
                        current_input = {"messages": [HumanMessage(content=user_response)]}
                        continue

                # No more work to do
                total_elapsed = time.perf_counter() - total_start_time
                logger.warning("RESEARCH COMPLETED WITHOUT FINAL REPORT")
                logger.warning("  result_keys: %s", list(result.keys()) if result else "None")
                logger.warning("  total_time: %.2fs", total_elapsed)
                
                console.print("[yellow]Research completed without final report.[/yellow]")
                if result:
                    console.print(f"[dim]Result keys: {list(result.keys())}[/dim]")
                break
            
            iteration_elapsed = time.perf_counter() - iteration_start
            logger.info("RUNNER ITERATION %d COMPLETE in %.2fs", iteration, iteration_elapsed)
                
        except Exception as e:
            total_elapsed = time.perf_counter() - total_start_time
            logger.error("RUN_RESEARCH ERROR in iteration %d after %.2fs: %s", iteration, total_elapsed, e)
            logger.exception("Full traceback:")
            console.print(f"\n[red]Error during research: {e}[/red]")
            import traceback
            traceback.print_exc()
            break
    
    if iteration >= max_iterations:
        total_elapsed = time.perf_counter() - total_start_time
        logger.warning("MAX ITERATIONS REACHED (%d) after %.2fs", max_iterations, total_elapsed)
        console.print("[red]Maximum iterations reached. Stopping.[/red]")
    
    return result if 'result' in dir() else {}


def main():
    """Main entry point"""
    logger.info("="*80)
    logger.info("DEEP RESEARCH RUNNER STARTED")
    logger.info("  Python: %s", sys.version.split()[0])
    logger.info("  Args: %s", sys.argv)
    
    console.print("\n[bold]╔═══════════════════════════════════════════════════════════════╗[/bold]")
    console.print("[bold]║    Pharmacy Newsletter - Interactive Research-to-Newsletter    ║[/bold]")
    console.print("[bold]╚═══════════════════════════════════════════════════════════════╝[/bold]\n")

    try:
        validate_pipeline_configuration()
    except Exception as exc:
        logger.error("CONFIGURATION PREFLIGHT FAILED: %s", exc)
        console.print(f"\n[red]Configuration preflight failed:[/red] {exc}")
        sys.exit(1)
    
    # Get research query / PDF
    source_filename = None  # Will be set if PDF is ingested
    pdf_path = None  # Will be set if PDF is provided
    
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        if Path(args[0]).exists() and args[0].lower().endswith(".pdf"):
            # Convert to absolute path for reliable access
            pdf_path = str(Path(args[0]).resolve())
            args = args[1:]
            # Extract filename without extension for output naming
            source_filename = Path(pdf_path).stem
            logger.info("  PDF path (absolute): %s", pdf_path)
            logger.info("  Source filename: %s", source_filename)

        query = " ".join(args) if args else ""

        # If PDF provided but no query, set default query
        if pdf_path and not query:
            query = "Build a pharmacy-leader newsletter from the provided article."

        if not query:
            logger.warning("No query provided, exiting")
            console.print("[red]No query provided. Exiting.[/red]")
            sys.exit(1)
        logger.info("  Query from args: %s", query[:100])
    else:
        # Interactive mode
        console.print("Enter the article/paper text or link for the pharmacy newsletter:")
        query = input("> ").strip()
        if not query:
            logger.warning("No query provided, exiting")
            console.print("[red]No query provided. Exiting.[/red]")
            sys.exit(1)
        logger.info("  Query from input: %s", query[:100])
    
    # Optional thread ID for resuming conversations
    thread_id = os.environ.get("THREAD_ID", "default")
    logger.info("  Thread ID: %s", thread_id)
    
    # Reset token tracker for fresh session
    reset_global_tracker()
    
    # Run the research
    session_start = time.perf_counter()
    try:
        result = asyncio.run(run_research(query, thread_id, source_filename, pdf_path))
        
        session_elapsed = time.perf_counter() - session_start
        
        # Log token usage summary
        tracker = get_global_tracker()
        tracker.log_summary(logger)
        
        exit_code = 0

        # Final status
        if result and "final_report" in result and result.get("final_report"):
            export_status = result.get("export_status") or (
                "structured_json" if result.get("structured_newsletter") else "failed"
            )
            structured_output_path = result.get("structured_output_path")
            raw_markdown_path = result.get("raw_markdown_path")
            export_error = result.get("export_error", "")

            if export_status == "structured_json":
                logger.info("="*80)
                logger.info("SESSION COMPLETED SUCCESSFULLY")
                logger.info("  total_session_time: %.2fs (%.1f minutes)", session_elapsed, session_elapsed/60)
                logger.info("  final_report_len: %d chars", len(result["final_report"]))
                logger.info("  structured_output_path: %s", structured_output_path)
                logger.info("  raw_markdown_path: %s", raw_markdown_path)
                logger.info("="*80)
                console.print("\n[bold green]✓ Newsletter completed successfully![/bold green]")
            elif export_status == "fallback_raw_markdown":
                exit_code = 1
                logger.warning("="*80)
                logger.warning("SESSION COMPLETED WITH STRUCTURED EXPORT FAILURE")
                logger.warning("  total_session_time: %.2fs (%.1f minutes)", session_elapsed, session_elapsed/60)
                logger.warning("  final_report_len: %d chars", len(result["final_report"]))
                logger.warning("  raw_markdown_path: %s", raw_markdown_path)
                logger.warning("  export_error: %s", export_error)
                logger.warning("="*80)
                console.print("\n[bold yellow]Newsletter generated, but structured JSON export failed.[/bold yellow]")
            else:
                exit_code = 1
                logger.error("="*80)
                logger.error("SESSION COMPLETED WITHOUT EXPORTED ARTIFACT")
                logger.error("  total_session_time: %.2fs (%.1f minutes)", session_elapsed, session_elapsed/60)
                logger.error("  final_report_len: %d chars", len(result["final_report"]))
                logger.error("  export_error: %s", export_error)
                logger.error("="*80)
                console.print("\n[bold red]Newsletter generated, but no export artifact was written.[/bold red]")

            if structured_output_path:
                console.print(f"\n[bold cyan]📄 Structured JSON exported to:[/bold cyan] {structured_output_path}")
            if raw_markdown_path:
                console.print(f"[bold cyan]📝 Raw markdown exported to:[/bold cyan] {raw_markdown_path}")
            if export_error:
                console.print(f"\n[yellow]Export detail:[/yellow] {export_error}")
            
            # Print token summary to console
            summary = tracker.get_summary()
            if summary["total_tokens"] > 0:
                console.print(f"\n[dim]Token usage: {summary['total_tokens']:,} total ({summary['total_prompt_tokens']:,} prompt + {summary['total_completion_tokens']:,} completion)[/dim]")
        else:
            logger.warning("SESSION ENDED WITHOUT FINAL REPORT")
            logger.warning("  total_session_time: %.2fs", session_elapsed)
            logger.warning("  result_keys: %s", list(result.keys()) if result else "None")
            console.print("\n[yellow]Research session ended.[/yellow]")

        if exit_code:
            sys.exit(exit_code)
            
    except KeyboardInterrupt:
        session_elapsed = time.perf_counter() - session_start
        logger.info("SESSION INTERRUPTED by user after %.2fs", session_elapsed)
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        session_elapsed = time.perf_counter() - session_start
        logger.error("SESSION ERROR after %.2fs: %s", session_elapsed, e)
        logger.exception("Full traceback:")
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
