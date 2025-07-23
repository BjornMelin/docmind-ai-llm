"""Predefined prompts, tones, instructions, and lengths for document analysis.

This module contains comprehensive configuration dictionaries that define the
available options for customizing document analysis behavior in DocMind AI.
It provides predefined prompts for common analysis tasks, tone specifications
for different writing styles, role-based instructions, and output length
preferences.

The configuration options allow users to tailor the analysis to their specific
needs, whether they require formal business analysis, academic research, or
creative interpretation of documents.

Example:
    Access predefined options::

        from prompts import PREDEFINED_PROMPTS, TONES

        prompt_text = PREDEFINED_PROMPTS["Comprehensive Document Analysis"]
        tone_instruction = TONES["Professional"]

Attributes:
    PREDEFINED_PROMPTS (dict): Mapping of prompt names to analysis instructions.
    TONES (dict): Mapping of tone names to style instructions.
    INSTRUCTIONS (dict): Mapping of role names to specialized instructions.
    LENGTHS (dict): Mapping of length preferences to detail instructions.

"""

PREDEFINED_PROMPTS = {
    "Comprehensive Document Analysis": (
        "Provide a summary, key insights, action items, and open questions "
        "from the document."
    ),
    "Extract Key Insights and Action Items": (
        "Extract key insights and action items from the document."
    ),
    "Summarize and Identify Open Questions": (
        "Summarize the document and identify open questions."
    ),
    "Custom Prompt": "",
}

TONES = {
    "Professional": "Use a professional, objective tone.",
    "Academic": "Use a scholarly, research-oriented tone.",
    "Informal": "Use a casual, conversational tone.",
    "Creative": "Use an imaginative, artistic tone.",
    "Neutral": "Use an unbiased, objective tone.",
    "Direct": "Use a concise, to-the-point tone.",
    "Empathetic": "Use an understanding, compassionate tone.",
    "Humorous": "Use a witty, lighthearted tone.",
    "Authoritative": "Use a confident, expert tone.",
    "Inquisitive": "Use a curious, exploratory tone.",
}

INSTRUCTIONS = {
    "General Assistant": "Act as a helpful assistant.",
    "Researcher": "Provide in-depth research and analysis.",
    "Software Engineer": "Focus on technical details and code.",
    "Product Manager": "Consider product strategy and user experience.",
    "Data Scientist": "Emphasize data analysis and modeling.",
    "Business Analyst": "Analyze from a business and strategic perspective.",
    "Technical Writer": "Create clear and concise documentation.",
    "Marketing Specialist": "Focus on branding and customer engagement.",
    "HR Manager": "Consider human resources aspects.",
    "Legal Advisor": "Provide information from a legal standpoint.",
    "Custom Instructions": "",
}

LENGTHS = {
    "Concise": "Keep the response brief and to-the-point.",
    "Detailed": "Provide a thorough and comprehensive response.",
    "Comprehensive": "Provide an extensive and in-depth response.",
    "Bullet Points": "Format the response in bullet points.",
}
