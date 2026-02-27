"""
Neuro-Lite Context Manager
───────────────────────────
Sliding window with bridge summary.
System prompt is NEVER evicted.
"""

import logging
import re
import hashlib
from typing import List, Dict, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("neurolite.context")

DEFAULT_MAX_TURNS = 20
BRIDGE_TRIGGER = 14


@dataclass
class ConversationSession:
    session_id: str
    system_prompt: str
    turns: List[Dict[str, str]] = field(default_factory=list)
    bridge_summary: Optional[str] = None
    max_turns: int = DEFAULT_MAX_TURNS

    def add_turn(self, role: str, content: str) -> None:
        self.turns.append({"role": role, "content": content})
        if len(self.turns) > self.max_turns:
            self._compact()

    def _compact(self) -> None:
        """Sliding window compaction with bridge summary."""
        if len(self.turns) <= BRIDGE_TRIGGER:
            return

        # Keep the last BRIDGE_TRIGGER turns, summarize older ones
        old_turns = self.turns[: -BRIDGE_TRIGGER]
        self.turns = self.turns[-BRIDGE_TRIGGER:]

        # Extract last topic from old turns (rule-based, no LLM)
        self.bridge_summary = self._extract_bridge(old_turns)
        logger.debug(
            "Context compacted: removed %d old turns, bridge='%s'",
            len(old_turns),
            (self.bridge_summary or "")[:80],
        )

    def _extract_bridge(self, old_turns: List[Dict[str, str]]) -> str:
        """
        Extract a bridge summary from old conversation turns.
        Rule-based: extract last user question topic.
        """
        last_user_msgs = [
            t["content"] for t in reversed(old_turns) if t["role"] == "user"
        ]
        if not last_user_msgs:
            return "The user was discussing a previous topic."

        last_msg = last_user_msgs[0]
        # Extract first sentence or first 120 chars
        first_sentence = re.split(r"[.!?\n]", last_msg)[0].strip()
        if len(first_sentence) > 120:
            first_sentence = first_sentence[:117] + "..."

        return f"Previously, the user was asking about: {first_sentence}"

    def build_messages(
        self,
        current_user_msg: str,
        emotion_modifier: str = "",
        rag_context: str = "",
    ) -> List[Dict[str, str]]:
        """
        Build the full message list for LLM inference.
        System prompt is always first and never removed.
        """
        messages = []

        # 1. System prompt (immutable)
        system_parts = [self.system_prompt]
        if emotion_modifier:
            system_parts.append(emotion_modifier)
        if rag_context:
            system_parts.append(
                f"\n[Reference Knowledge]\n{rag_context}\n"
                "Use the above reference if relevant. "
                "If the reference doesn't apply, ignore it."
            )
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

        # 2. Bridge summary (if compaction occurred)
        if self.bridge_summary:
            messages.append({
                "role": "system",
                "content": f"[Context Bridge] {self.bridge_summary}",
            })

        # 3. Conversation history (sliding window)
        for turn in self.turns:
            messages.append(turn)

        # 4. Current user message
        messages.append({"role": "user", "content": current_user_msg})

        return messages

    def get_turn_count(self) -> int:
        return len(self.turns)


class ContextManager:
    """
    Manages multiple conversation sessions.
    Thread-safe via dict isolation per session.
    """

    def __init__(self, system_prompt: str, max_turns: int = DEFAULT_MAX_TURNS):
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._sessions: Dict[str, ConversationSession] = {}
        self._max_sessions = 200  # Prevent memory leak

    def get_session(self, session_id: str) -> ConversationSession:
        if session_id not in self._sessions:
            if len(self._sessions) >= self._max_sessions:
                self._evict_oldest()
            self._sessions[session_id] = ConversationSession(
                session_id=session_id,
                system_prompt=self._system_prompt,
                max_turns=self._max_turns,
            )
        return self._sessions[session_id]

    def delete_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def _evict_oldest(self) -> None:
        """Remove oldest 20% of sessions."""
        to_remove = max(1, len(self._sessions) // 5)
        keys = list(self._sessions.keys())[:to_remove]
        for k in keys:
            del self._sessions[k]
        logger.info("Evicted %d old sessions.", to_remove)

    @staticmethod
    def generate_session_id(client_info: str = "") -> str:
        import time
        raw = f"{time.time_ns()}-{client_info}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
