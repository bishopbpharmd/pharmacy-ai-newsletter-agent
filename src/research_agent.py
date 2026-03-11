
"""Research Agent Implementation.

This module implements a research agent that can perform iterative web searches
and synthesis to answer complex research questions.
"""

import time
from typing_extensions import Literal

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage, AIMessage, filter_messages

from src.model_config import (
    PIPELINE_MODEL_SETTINGS,
    build_chat_model,
)
from src.state_research import ResearcherState, ResearcherOutputState
from src.utils import tavily_search, get_today_str, think_tool, retrieve_document_chunks, DEFAULT_DOC_ID
from src.prompts import research_agent_prompt, compress_research_system_prompt, compress_research_human_message
from src.logging_config import get_logger, summarize_message, log_token_usage, get_global_tracker

# ===== LOGGING SETUP =====

logger = get_logger("deep_research.research_agent")

# ===== CONFIGURATION =====

# Set up tools and model binding
tools = [tavily_search, think_tool, retrieve_document_chunks]
tools_by_name = {tool.name: tool for tool in tools}

# Initialize models
model = build_chat_model(PIPELINE_MODEL_SETTINGS.research_agent_tool_call_model)
model_with_tools = model.bind_tools(tools)
summarization_model = build_chat_model(PIPELINE_MODEL_SETTINGS.research_agent_summarization_model)
compress_model = build_chat_model(PIPELINE_MODEL_SETTINGS.research_agent_compression_model)

logger.info("Research agent initialized with tools: %s", [t.name for t in tools])

# ===== AGENT NODES =====

def llm_call(state: ResearcherState):
    """Analyze current state and decide on next actions.

    The model analyzes the current conversation state and decides whether to:
    1. Call search tools to gather more information
    2. Provide a final answer based on gathered information

    Returns updated state with the model's response.
    """
    research_topic = state.get("research_topic", "Unknown")
    messages = state.get("researcher_messages", [])
    msg_count = len(messages)
    
    start_time = time.perf_counter()
    
    logger.info("="*60)
    logger.info("LLM_CALL START | topic='%s' | message_count=%d", research_topic[:100], msg_count)
    
    # Calculate and log context size
    total_content_size = sum(len(str(getattr(msg, 'content', ''))) for msg in messages)
    logger.info("  total_context_size: %d chars (%.1f KB)", total_content_size, total_content_size/1024)
    
    # Log existing messages summary
    for i, msg in enumerate(messages):
        logger.info("  existing_msg[%d]: %s", i, summarize_message(msg))
    
    # Format the prompt with the current date
    formatted_prompt = research_agent_prompt.format(date=get_today_str(), doc_id=DEFAULT_DOC_ID)
    
    llm_start = time.perf_counter()
    try:
        response = model_with_tools.invoke(
            [SystemMessage(content=formatted_prompt)] + state["researcher_messages"]
        )
        llm_elapsed = time.perf_counter() - llm_start
        
        # Log the response and token usage
        tool_calls = getattr(response, 'tool_calls', [])
        log_token_usage(logger, response, "research_agent_llm")
        get_global_tracker().add_usage(response, "research_agent_llm")
        
        logger.info("LLM_CALL RESPONSE | tool_calls=%d | content_len=%d | llm_time=%.2fs", 
                   len(tool_calls) if tool_calls else 0, 
                   len(str(response.content)),
                   llm_elapsed)
        
        if tool_calls:
            for tc in tool_calls:
                logger.info("  tool_call: %s (id=%s)", tc.get('name'), tc.get('id'))
                # Log search query if it's a tavily_search
                if tc.get('name') == 'tavily_search':
                    query = tc.get('args', {}).get('query', '')[:80]
                    logger.info("    query: %s...", query)
        else:
            logger.info("  NO TOOL CALLS - LLM decided to provide final answer")
            logger.info("  response_preview: %s", str(response.content)[:300])
        
        total_elapsed = time.perf_counter() - start_time
        logger.info("LLM_CALL COMPLETE | total_time=%.2fs", total_elapsed)
        
        return {"researcher_messages": [response]}
        
    except Exception as e:
        llm_elapsed = time.perf_counter() - llm_start
        logger.error("LLM_CALL ERROR after %.2fs: %s", llm_elapsed, e)
        logger.exception("Full traceback:")
        raise

def tool_node(state: ResearcherState):
    """Execute all tool calls from the previous LLM response.

    Executes all tool calls from the previous LLM responses.
    Returns updated state with tool execution results.
    """
    start_time = time.perf_counter()
    
    tool_calls = state["researcher_messages"][-1].tool_calls
    logger.info("="*60)
    logger.info("TOOL_NODE START | executing %d tool calls", len(tool_calls))

    # Execute all tool calls
    observations = []
    total_result_chars = 0
    
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        logger.info("  executing tool[%d]: %s", i, tool_name)
        logger.info("    args: %s", str(tool_args)[:200])
        
        tool_start = time.perf_counter()
        try:
            tool = tools_by_name[tool_name]
            result = tool.invoke(tool_args)
            tool_elapsed = time.perf_counter() - tool_start
            
            observations.append(result)
            result_len = len(str(result))
            total_result_chars += result_len
            logger.info("    result_len: %d chars in %.2fs", result_len, tool_elapsed)
            
            # For tavily_search, log more detail about result quality
            if tool_name == "tavily_search":
                # Check if result looks like it has actual content
                if result_len < 100:
                    logger.warning("    TAVILY SEARCH WARNING - very small result (%d chars)", result_len)
                elif result_len < 500:
                    logger.info("    TAVILY SEARCH - minimal results (%d chars)", result_len)
                else:
                    logger.info("    TAVILY SEARCH - got %d chars of results", result_len)
                    
        except Exception as e:
            tool_elapsed = time.perf_counter() - tool_start
            logger.error("    TOOL ERROR after %.2fs: %s", tool_elapsed, e)
            logger.exception("    Full traceback:")
            observations.append(f"Error executing {tool_name}: {str(e)}")

    # Create tool message outputs
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"]
        ) for observation, tool_call in zip(observations, tool_calls)
    ]
    
    total_elapsed = time.perf_counter() - start_time
    logger.info("TOOL_NODE COMPLETE | messages=%d | total_chars=%d | time=%.2fs", 
               len(tool_outputs), total_result_chars, total_elapsed)

    return {"researcher_messages": tool_outputs}

def build_clean_messages_for_compression(messages: list) -> list:
    """Build a clean message list for the compression model (which has NO tools bound).
    
    The compression model is NOT bound to tools, so we convert:
    1. HumanMessages → kept as-is
    2. AIMessages → converted to plain text (strip tool_calls)
    3. tavily_search ToolMessages → converted to HumanMessages with search results
    4. think_tool ToolMessages → skipped (internal reasoning)
    """
    clean_messages = []
    
    for msg in messages:
        if isinstance(msg, HumanMessage):
            clean_messages.append(msg)
        
        elif isinstance(msg, AIMessage):
            # Convert to plain AIMessage - strip tool_calls
            if msg.content:
                clean_messages.append(AIMessage(content=msg.content))
            elif hasattr(msg, 'tool_calls') and msg.tool_calls:
                tool_names = [tc.get('name', 'unknown') for tc in msg.tool_calls]
                clean_messages.append(AIMessage(content=f"[Executing: {', '.join(tool_names)}]"))
        
        elif isinstance(msg, ToolMessage):
            # Convert tavily_search results to labeled HumanMessages
            if getattr(msg, 'name', '') == 'tavily_search':
                clean_messages.append(HumanMessage(content=f"[Search Results]\n{msg.content}"))
    
    return clean_messages

def compress_research(state: ResearcherState) -> dict:
    """Compress research findings into a concise summary.

    Takes all the research messages and tool outputs and creates
    a compressed summary suitable for the supervisor's decision-making.
    """
    start_time = time.perf_counter()
    
    research_topic = state.get("research_topic", "Unknown topic")
    researcher_messages = state.get("researcher_messages", [])
    
    logger.info("="*60)
    logger.info("COMPRESS_RESEARCH START | topic='%s'", research_topic[:100])
    logger.info("  total_messages: %d", len(researcher_messages))
    
    # Count message types
    human_count = sum(1 for m in researcher_messages if isinstance(m, HumanMessage))
    ai_count = sum(1 for m in researcher_messages if isinstance(m, AIMessage))
    tool_count = sum(1 for m in researcher_messages if isinstance(m, ToolMessage))
    tavily_count = sum(1 for m in researcher_messages if isinstance(m, ToolMessage) and getattr(m, 'name', '') == 'tavily_search')
    
    # Calculate total content size
    total_content_size = sum(len(str(getattr(m, 'content', ''))) for m in researcher_messages)
    
    logger.info("  message_breakdown: human=%d, ai=%d, tool=%d (tavily=%d)", 
               human_count, ai_count, tool_count, tavily_count)
    logger.info("  total_content_size: %d chars (%.1f KB)", total_content_size, total_content_size/1024)
    
    if tavily_count == 0:
        logger.warning("  WARNING: No tavily_search results found! Research may not have actual web data.")

    # Build clean messages for compression model (strips tool_calls)
    clean_messages = build_clean_messages_for_compression(researcher_messages)
    clean_content_size = sum(len(str(getattr(m, 'content', ''))) for m in clean_messages)
    logger.info("  clean_messages_for_compression: %d (%.1f KB)", len(clean_messages), clean_content_size/1024)
    
    system_message = compress_research_system_prompt.format(date=get_today_str())
    human_message = compress_research_human_message.format(
        research_topic=research_topic,
        perspectives_with_plans="(single-agent research; no distinct perspective plans)",
    )
    
    messages = [SystemMessage(content=system_message)] + clean_messages + [HumanMessage(content=human_message)]
    
    llm_start = time.perf_counter()
    try:
        response = compress_model.invoke(messages)
        llm_elapsed = time.perf_counter() - llm_start
        
        compressed_len = len(str(response.content))
        compression_ratio = total_content_size / compressed_len if compressed_len > 0 else 0
        
        # Log token usage
        log_token_usage(logger, response, "compress_research")
        get_global_tracker().add_usage(response, "compress_research")
        
        logger.info("COMPRESS_RESEARCH COMPLETE")
        logger.info("  compressed_len: %d chars", compressed_len)
        logger.info("  compression_ratio: %.1fx (%d -> %d)", compression_ratio, total_content_size, compressed_len)
        logger.info("  llm_time: %.2fs", llm_elapsed)
        
    except Exception as e:
        llm_elapsed = time.perf_counter() - llm_start
        logger.error("COMPRESS_RESEARCH ERROR after %.2fs: %s", llm_elapsed, e)
        logger.exception("Full traceback:")
        raise

    # Extract raw notes from tavily_search tool messages only
    raw_notes = [
        str(m.content) for m in researcher_messages
        if isinstance(m, ToolMessage) and getattr(m, 'name', '') == 'tavily_search'
    ]
    raw_notes_chars = sum(len(n) for n in raw_notes)
    logger.info("  raw_notes: count=%d, total_chars=%d", len(raw_notes), raw_notes_chars)
    
    total_elapsed = time.perf_counter() - start_time
    logger.info("  total_compress_time: %.2fs", total_elapsed)

    return {
        "compressed_research": str(response.content),
        "raw_notes": ["\n".join(raw_notes)]
    }

# ===== ROUTING LOGIC =====

def should_continue(state: ResearcherState) -> Literal["tool_node", "compress_research"]:
    """Determine whether to continue research or provide final answer.

    Determines whether the agent should continue the research loop or provide
    a final answer based on whether the LLM made tool calls.

    Returns:
        "tool_node": Continue to tool execution
        "compress_research": Stop and compress research
    """
    messages = state["researcher_messages"]
    last_message = messages[-1]
    
    has_tool_calls = bool(getattr(last_message, 'tool_calls', None))
    
    if has_tool_calls:
        tool_names = [tc.get('name', '?') for tc in last_message.tool_calls]
        logger.info("SHOULD_CONTINUE: has_tool_calls=True -> tool_node (tools: %s)", tool_names)
        return "tool_node"
    else:
        logger.info("SHOULD_CONTINUE: has_tool_calls=False -> compress_research")
        return "compress_research"

# ===== GRAPH CONSTRUCTION =====

# Build the agent workflow
agent_builder = StateGraph(ResearcherState, output_schema=ResearcherOutputState)

# Add nodes to the graph
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_node("compress_research", compress_research)

# Add edges to connect nodes
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call",
    should_continue,
    {
        "tool_node": "tool_node", # Continue research loop
        "compress_research": "compress_research", # Provide final answer
    },
)
agent_builder.add_edge("tool_node", "llm_call") # Loop back for more research
agent_builder.add_edge("compress_research", END)

# Compile the agent
researcher_agent = agent_builder.compile()
