"""
Neuro-Lite Emotional State Classifier
──────────────────────────────────────
Pure regex + heuristic scoring. NO ML models.
Target: <1ms execution.
States: neutral | concerned | celebratory | frustrated | curious | grateful
Returns persona modifier string for prompt injection.
"""

import re
import logging
from dataclasses import dataclass
from typing import Tuple

logger = logging.getLogger("neurolite.emotion")

# ═══════════════════════════════════════════════════════════
# Pattern Definitions
# ═══════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class EmotionSignal:
    patterns: Tuple[re.Pattern, ...]
    weight: float


CONCERN_PATTERNS = EmotionSignal(
    patterns=tuple(re.compile(p, re.IGNORECASE) for p in [
        r"\b(help|please|urgent|emergency|critical|broken|fail|error|crash|die|dead|lost|stuck)\b",
        r"\b(can'?t|cannot|unable|impossible|won'?t|doesn'?t work|not working)\b",
        r"\b(worried|anxious|scared|afraid|terrified|panic|stress|depress)\b",
        r"\b(problem|issue|bug|trouble|wrong|bad|terrible|horrible|worst)\b",
        r"\b(sick|ill|pain|hurt|suffer|bleed|hospital|doctor)\b",
        r"[\?]{2,}",
        r"[!]{3,}",
    ]),
    weight=1.0,
)

CELEBRATION_PATTERNS = EmotionSignal(
    patterns=tuple(re.compile(p, re.IGNORECASE) for p in [
        r"\b(thank|thanks|thx|awesome|amazing|great|excellent|perfect|wonderful|love)\b",
        r"\b(congratulat|celebrate|success|achieved|won|winner|happy|glad|excited)\b",
        r"\b(yay|hooray|woo+|yes+|finally|brilliant)\b",
        r"[🎉🎊🥳😊😃👍❤️💯🔥✅🙏]+",
        r"\b(solved|fixed|working now|it works|got it)\b",
    ]),
    weight=1.0,
)

FRUSTRATION_PATTERNS = EmotionSignal(
    patterns=tuple(re.compile(p, re.IGNORECASE) for p in [
        r"\b(wtf|damn|shit|fuck|crap|stupid|dumb|useless|waste|hate|suck)\b",
        r"\b(annoying|annoyed|frustrated|irritat|furious|angry|mad)\b",
        r"\b(again|still|already|yet another|every time|always breaks)\b",
        r"[!]{2,}",
        r"\b(ugh+|argh+|grr+)\b",
    ]),
    weight=1.2,
)

CURIOSITY_PATTERNS = EmotionSignal(
    patterns=tuple(re.compile(p, re.IGNORECASE) for p in [
        r"\b(how|why|what|when|where|which|who|explain|curious|wonder|learn)\b",
        r"\b(tell me|can you|could you|would you|is it possible|teach)\b",
        r"\b(understand|meaning|definition|difference|compare|versus)\b",
        r"\b(example|demonstrate|show me|walk me through)\b",
    ]),
    weight=0.6,
)

GRATITUDE_PATTERNS = EmotionSignal(
    patterns=tuple(re.compile(p, re.IGNORECASE) for p in [
        r"\b(thank|thanks|thx|appreciate|grateful|gratitude)\b",
        r"\b(you'?re the best|so helpful|really helps|saved my)\b",
        r"[🙏❤️💕]+",
    ]),
    weight=1.1,
)

# ═══════════════════════════════════════════════════════════
# Persona Modifiers
# ═══════════════════════════════════════════════════════════

PERSONA_MODIFIERS = {
    "neutral": "",
    "concerned": "The user seems worried or facing a problem. Respond calmly and reassuringly. Be supportive and solution-oriented. Show empathy first, then provide help.",
    "celebratory": "The user is happy or celebrating something positive. Match their positive energy warmly. Acknowledge their achievement or good mood.",
    "frustrated": "The user appears frustrated. Stay patient, empathetic, and professional. Acknowledge their frustration without being dismissive. Focus on actionable solutions.",
    "curious": "The user is curious and wants to learn. Be thorough and educational. Use clear explanations with examples where helpful.",
    "grateful": "The user is expressing gratitude. Accept it graciously and warmly. Reinforce your willingness to help further.",
}


def classify_emotion(text: str) -> Tuple[str, str, float]:
    """
    Classify emotional state of input text.

    Returns:
        Tuple of (emotion_label, persona_modifier, confidence)
    """
    if not text or not text.strip():
        return "neutral", PERSONA_MODIFIERS["neutral"], 0.0

    scores = {
        "concerned": 0.0,
        "celebratory": 0.0,
        "frustrated": 0.0,
        "curious": 0.0,
        "grateful": 0.0,
    }

    signal_map = {
        "concerned": CONCERN_PATTERNS,
        "celebratory": CELEBRATION_PATTERNS,
        "frustrated": FRUSTRATION_PATTERNS,
        "curious": CURIOSITY_PATTERNS,
        "grateful": GRATITUDE_PATTERNS,
    }

    text_lower = text.lower()
    text_len = max(len(text_lower.split()), 1)

    for emotion, signal in signal_map.items():
        match_count = 0
        for pattern in signal.patterns:
            matches = pattern.findall(text_lower)
            match_count += len(matches)
        # Normalize by text length to avoid bias toward long messages
        scores[emotion] = (match_count * signal.weight) / (text_len ** 0.5)

    best_emotion = max(scores, key=scores.get)
    best_score = scores[best_emotion]

    # Threshold: if no strong signal, return neutral
    if best_score < 0.3:
        return "neutral", PERSONA_MODIFIERS["neutral"], 0.0

    confidence = min(best_score / 3.0, 1.0)

    logger.debug(
        "Emotion classified: %s (confidence=%.2f, scores=%s)",
        best_emotion, confidence, {k: round(v, 2) for k, v in scores.items()}
    )

    return best_emotion, PERSONA_MODIFIERS[best_emotion], confidence
