"""
Neuro-Lite Post Processor
─────────────────────────
Rule-based response polishing.
Target: <2ms execution.
NO LLM re-inference. String manipulation only.
"""

import re
import time
import logging
from typing import Optional

logger = logging.getLogger("neurolite.postproc")

# ═══════════════════════════════════════════════════════════
# Empathy Prefix Templates
# ═══════════════════════════════════════════════════════════

EMPATHY_PREFIX = {
    "concerned": [
        "I understand this can be stressful. ",
        "I hear your concern. ",
        "Let me help you with this. ",
    ],
    "frustrated": [
        "I understand your frustration. ",
        "That does sound frustrating — let's sort this out. ",
        "I'm sorry you're dealing with this. ",
    ],
    "celebratory": [
        "That's wonderful! ",
        "Great to hear! ",
        "Fantastic! ",
    ],
    "grateful": [
        "You're very welcome! ",
        "Happy to help! ",
        "Glad I could assist! ",
    ],
}

# Track which prefix was last used per emotion to rotate
_last_prefix_idx: dict = {}


def _get_empathy_prefix(emotion: str) -> str:
    """Rotate through empathy prefixes to avoid repetition."""
    if emotion not in EMPATHY_PREFIX:
        return ""
    options = EMPATHY_PREFIX[emotion]
    idx = _last_prefix_idx.get(emotion, -1) + 1
    if idx >= len(options):
        idx = 0
    _last_prefix_idx[emotion] = idx
    return options[idx]


# ═══════════════════════════════════════════════════════════
# Text Cleaning Rules
# ═══════════════════════════════════════════════════════════

# Compiled once at import time for speed
_RE_MULTI_NEWLINES = re.compile(r"\n{4,}")
_RE_MULTI_SPACES = re.compile(r"[ \t]{3,}")
_RE_TRAILING_SPACES = re.compile(r"[ \t]+$", re.MULTILINE)
_RE_ORPHAN_BULLETS = re.compile(r"^\s*[-*•]\s*$", re.MULTILINE)
_RE_REPEATED_PUNCT = re.compile(r"([.!?])\1{2,}")
_RE_BROKEN_SENTENCE = re.compile(r"\s+([,.:;!?])")
_RE_HALLUCINATION_DISCLAIMER = re.compile(
    r"(as an ai|i'?m (?:just )?an? (?:ai|language model|llm)|i don'?t have (?:feelings|emotions|consciousness))",
    re.IGNORECASE,
)


def polish_text(text: str) -> str:
    """
    Clean up LLM output formatting.
    Must complete in <2ms.
    """
    if not text:
        return text

    result = text.strip()

    # Fix excessive newlines
    result = _RE_MULTI_NEWLINES.sub("\n\n\n", result)

    # Fix excessive spaces
    result = _RE_MULTI_SPACES.sub("  ", result)

    # Remove trailing whitespace per line
    result = _RE_TRAILING_SPACES.sub("", result)

    # Remove orphaned bullet points
    result = _RE_ORPHAN_BULLETS.sub("", result)

    # Fix repeated punctuation
    result = _RE_REPEATED_PUNCT.sub(r"\1", result)

    # Fix broken punctuation spacing
    result = _RE_BROKEN_SENTENCE.sub(r"\1", result)

    # Remove potential empty lines at start
    result = result.lstrip("\n")

    return result


def compute_prefix(emotion: str, response_text: str) -> str:
    """
    Compute empathy prefix if needed.
    Checks that we don't duplicate empathy already present in text.
    """
    if emotion in ("neutral", "curious"):
        return ""

    prefix = _get_empathy_prefix(emotion)
    if not prefix:
        return ""

    # Check if the LLM already started empathetically
    first_40 = response_text[:80].lower() if response_text else ""
    empathy_indicators = [
        "understand", "sorry", "hear you", "frustrat", "wonderful",
        "great to", "fantastic", "welcome", "glad", "happy to help",
        "concern", "stressful", "that's great", "congrat",
    ]
    for indicator in empathy_indicators:
        if indicator in first_40:
            return ""  # LLM already handled it

    return prefix


def compute_suffix(full_response: str, emotion: str) -> str:
    """
    Compute professional closing suffix if needed.
    Only add if response doesn't already end with a closing.
    """
    if not full_response:
        return ""

    last_100 = full_response[-100:].lower()
    closing_indicators = [
        "let me know", "feel free", "happy to help",
        "hope this helps", "don't hesitate", "any questions",
        "anything else", "further assistance", "glad to assist",
    ]
    for ind in closing_indicators:
        if ind in last_100:
            return ""

    # Only add closing for longer responses (>200 chars) when user is concerned/frustrated
    if emotion in ("concerned", "frustrated") and len(full_response) > 200:
        return "\n\nFeel free to ask if you need further help."

    return ""


def process_response(
    text: str,
    emotion: str = "neutral",
    confidence: float = 0.0,
) -> str:
    """
    Full post-processing pipeline.
    Must execute in <2ms total.
    """
    start = time.monotonic()

    # 1. Polish text formatting
    polished = polish_text(text)

    # 2. Add empathy prefix (if not already present)
    if confidence > 0.3:
        prefix = compute_prefix(emotion, polished)
        if prefix:
            polished = prefix + polished

        suffix = compute_suffix(polished, emotion)
        if suffix:
            polished = polished + suffix

    elapsed_us = (time.monotonic() - start) * 1_000_000
    if elapsed_us > 2000:
        logger.warning("Post-processor exceeded 2ms: %.0fµs", elapsed_us)

    return polished
