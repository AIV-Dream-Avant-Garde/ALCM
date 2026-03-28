"""Context type detection from conversational signals.

Detects situational context (professional, casual, intimate, formal,
conflict, creative, public) from input keywords and conversation history.
See Developer Guide Section 4.6.
"""
from typing import Optional

# Context type keywords for detection
CONTEXT_KEYWORDS = {
    "PROFESSIONAL": ["meeting", "business", "quarterly", "revenue", "client", "deadline", "strategy"],
    "CASUAL": ["hey", "lol", "haha", "chill", "hang out", "weekend", "fun"],
    "FORMAL": ["dear", "sincerely", "respectfully", "hereby", "pursuant"],
    "CONFLICT": ["disagree", "wrong", "frustrated", "disappointed", "unacceptable", "complaint"],
    "CREATIVE": ["brainstorm", "imagine", "what if", "idea", "create", "design", "concept"],
    "INTIMATE": ["love", "miss you", "feeling", "worried about us", "relationship"],
}


def detect_context_type(context: str, conversation_history: Optional[list] = None) -> str:
    """Detect situational context type from input signals.

    Returns: PROFESSIONAL | CASUAL | INTIMATE | FORMAL | CONFLICT | CREATIVE | PUBLIC
    """
    text = context.lower()
    if conversation_history:
        for msg in conversation_history[-3:]:
            text += " " + (msg.get("content", "") or "").lower()

    scores = {}
    for ctx_type, keywords in CONTEXT_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text)
        if score > 0:
            scores[ctx_type] = score

    if scores:
        return max(scores, key=scores.get)
    return "PUBLIC"
