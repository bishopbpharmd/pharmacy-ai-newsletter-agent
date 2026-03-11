"""
Newsletter Templates Module

This module contains template definitions for different newsletter types.
Templates are stored as structured data and can be dynamically injected
into prompts at runtime.
"""

from typing import Dict, List, Optional

# ===== TEMPLATE DEFINITIONS =====

NEWSLETTER_TEMPLATES: Dict[str, dict] = {
    "research_article": {
        "name": "Research Article Template",
        "description": "For formal studies with methods, data, outcomes, and limitations",
        "selection_criteria": "Use when the source is a formal study (methods, sample size, data, outcomes, limitations) or clearly labeled as a trial/retrospective/observational study",
        "research_guidance": {
            "retrieval_focus": "Use retrieve_document_chunks with natural-language questions to pull: study design and setting; cohort size and inclusion/exclusion; date range; key outcomes and effect sizes; limitations; comparators or baseline care.",
            "external_research": "Use external search selectively for unresolved external gaps only: related clinical guidelines, similar studies, validation data, artifact discovery, or regulatory context that materially changes interpretation. Keep external content complementary to the article, not automatic.",
            "content_balance": "article-dominant; external context only when it materially improves interpretation or actionability",
            "max_research_cycles": 3
        },
        "sections": [
            {
                "header": "Quick Take",
                "guidance": """Synthesize the study into two high-impact bullets designed for a busy inpatient hospital pharmacy leader:
- Focus: Combine the primary intervention/result (must include quantitative data/metrics) with the practical 'so what' for pharmacy operations or safety.
- Constraint: Zero background or fluff. Assume the reader knows the background and just wants the bottom line."""
            },
            {
                "header": "Why It Matters",
                "guidance": """Draft 2–3 bullets on the Problem Landscape & Urgency. For example, but not limited to:
- Articulate the specific clinical gap or operational bottleneck (friction) this study addresses (the immediate pain point).
- Explain why this gap is unsustainable—connect it to broader pharmacy challenges. For example (not an exhaustive list): resource constraints, safety risks, clinician burnout, etc.
- Sell the problem, not the solution. Focus entirely on the 'Before' state and the cost of inaction; do not mention the study's results."""
            },
            {
                "header": "What They Did",
                "guidance": """Outline the Methodology in 3–4 plain-language bullets (avoid data science jargon). For example, but not limited to:
- Define the setting and population (who & where), explicitly noting any design details that affect transferability (e.g., single-center vs. multi-system).
- Describe the specific data inputs (EHR notes, labs, meds) and the core intervention.
- Explain how the tool was tested—specifically distinguishing between a theoretical retrospective study and a live clinical deployment."""
            },
            {
                "header": "What They Found",
                "guidance": """List the Key Results in 3–5 bullets. For example, but not limited to:
- Present the primary quantitative outcomes (accuracy, time saved, error signals) using exact metrics as reported; do not extrapolate or round up.
- Provide a calibrated interpretation of the signal strength, explicitly noting any limitations like retrospective design, sample size, or single-center bias.
- Connect the results directly to pharmacy practice, translating abstract numbers into operational terms (e.g., specific impact on verification time, stewardship interventions, or alert burden).
- Maintain a skeptical but fair tone—distinguish clearly between theoretical model performance (AUC/F1 scores) and proven operational gains."""
            },
            {
                "header": "What This Means for Us",
                "guidance": """What This Means for Us (3–4 Strategic Bullets). For example, but not limited to:
- Classify the tool’s immediate utility—is this a "triage co-pilot" for high-volume queues, or a "second set of eyes" for high-risk patients?
- A direct "For Pharmacists" line explaining how this changes the daily cognitive load (e.g., "Shifts the focus from data gathering to judgment").
- What specific question should a leader ask a vendor or IT team before building or looking into vendors for this? (Focus on data fit, maintenance costs, or specific blind spots).
- Avoid generic calls for local validation. Instead, identify the specific clinical scenario where this model is most likely to fail or requires the heaviest human oversight."""
            },
            {
                "header": "Strengths & Limitations",
                "guidance": """Present strengths and limitations together in a nested bullet structure:
  - **Strengths**
    - 1–2 bullets on what this work does well (for example, but not limited to: real-world data, prospective design, transparent reporting, workflow integration, etc.).
  - **Limitations**
    - 1–2 bullets on constraints that matter (for example, but not limited to: single-center, narrow population, short follow-up, missing outcomes, limited deployment details, vendor opacity).
- Maintain a matter-of-fact, non-defensive tone."""
            },
            {
                "header": "Bottom Line",
                "guidance": """Single, high-impact sentence (roughly 25 words or fewer).
- Answer: "How should a pharmacy leader mentally file this study and what should they watch for in similar tools?"
- Use calibrated language: "early signal", "promising for triage", "hypothesis-generating", rather than "ready for widespread deployment"."""
            }
        ]
    },
    "commentary": {
        "name": "Commentary/News Template",
        "description": "For opinion pieces, news, policy updates, and perspectives",
        "selection_criteria": "Use for opinion pieces, news articles, policy updates, perspectives, or any non-research content",
        "research_guidance": {
            "retrieval_focus": "case examples, deployment story, policy implications, argument claims, stakeholder reactions",
            "external_research": "optional — only if critical context is missing (e.g., regulatory status, policy background)",
            "content_balance": "~80-90% article content, external research fills small gaps only",
            "max_research_cycles": 2
        },
        "sections": [
            {
                "header": "Quick Take",
                "guidance": """- 2–3 bullets with the core change/claim and who's involved.
- Note whether this is live, piloted, or speculative.
- Briefly highlight the main tension for inpatient pharmacy."""
            },
            {
                "header": "Key Details",
                "guidance": """- 2–4 bullets describing what actually happened or was argued.
- Concrete features, mechanisms, examples; factual, not advisory."""
            },
            {
                "header": "Why it matters",
                "guidance": """- 2–4 bullets interpreting relevance for inpatient pharmacy, operations, safety, or governance.
- Use conditional language ("could", "may", "raises questions about")."""
            },
            {
                "header": "Bottom Line",
                "guidance": """- 1–2 sentences with a neutral, executive-ready frame and mental label (e.g., "early signal", "watchlist")."""
            }
        ]
    }
}


# ===== HELPER FUNCTIONS =====

def get_template(template_name: str) -> dict:
    """Get a template by name.
    
    Args:
        template_name: The template identifier (e.g., "research_article", "commentary")
        
    Returns:
        The template dictionary
        
    Raises:
        ValueError: If template_name is not found
    """
    if template_name not in NEWSLETTER_TEMPLATES:
        available = ", ".join(NEWSLETTER_TEMPLATES.keys())
        raise ValueError(f"Unknown template '{template_name}'. Available templates: {available}")
    return NEWSLETTER_TEMPLATES[template_name]


def get_template_sections_list(template_name: str) -> List[str]:
    """Get list of section headers for a template.
    
    Args:
        template_name: The template identifier
        
    Returns:
        List of section header strings
    """
    template = get_template(template_name)
    return [section["header"] for section in template["sections"]]


def get_available_templates_info() -> str:
    """Return formatted markdown listing all available templates for selection prompt.
    
    Returns:
        Formatted markdown string with template names, descriptions, and selection criteria
    """
    lines = ["**Available Templates:**\n"]
    
    for i, (key, template) in enumerate(NEWSLETTER_TEMPLATES.items(), 1):
        lines.append(f"{i}. **{key}**")
        lines.append(f"   - Description: {template['description']}")
        lines.append(f"   - When to use: {template['selection_criteria']}")
        lines.append(f"   - Sections: {', '.join(get_template_sections_list(key))}")
        lines.append("")
    
    return "\n".join(lines)


def format_template_for_prompt(template_name: str) -> str:
    """Format template as markdown string ready for prompt injection.
    
    Args:
        template_name: The template identifier
        
    Returns:
        Formatted markdown string matching the template format expected by prompts
    """
    template = get_template(template_name)
    
    lines = [f"### {template['name']}\n"]
    lines.append("# Title")
    
    # Add title guidance for research article
    if template_name == "research_article":
        lines.append("- Clear, non-clickbait; reflect the main tool/outcome or question for pharmacy.")
    elif template_name == "commentary":
        lines.append("- Clear, non-clickbait")
    
    lines.append("")
    
    # Add each section
    for section in template["sections"]:
        lines.append(f"## {section['header']}")
        lines.append(section["guidance"])
        lines.append("")
    
    # Add section adherence instruction
    section_list = ", ".join(get_template_sections_list(template_name))
    lines.append("---\n")
    lines.append(f"**CRITICAL - Section Template Adherence**: You MUST use ONLY the exact section headers from this template: {section_list}. Do NOT add new sections or modify section names.")
    lines.append("")
    lines.append('**Output headers**: Use clean section names only (e.g., do NOT include the number of bullets (2–3 bullets)).')
    
    return "\n".join(lines)


def get_valid_template_names() -> List[str]:
    """Return list of valid template names.
    
    Returns:
        List of template name strings
    """
    return list(NEWSLETTER_TEMPLATES.keys())


def get_research_guidance(template_name: str) -> str:
    """Format research guidance for prompt injection.
    
    Args:
        template_name: The template identifier
        
    Returns:
        Formatted markdown string with research priorities for the template type
    """
    template = get_template(template_name)
    guidance = template.get("research_guidance", {})
    
    if not guidance:
        return ""
    
    lines = [
        f"**Research Guidance for {template['name']}**:",
        f"- Retrieval focus: {guidance.get('retrieval_focus', '')}",
        f"- External research: {guidance.get('external_research', '')}",
        f"- Content balance: {guidance.get('content_balance', '')}",
        f"- Max research cycles: {guidance.get('max_research_cycles', 2)}"
    ]
    
    return "\n".join(lines)
