"""
Dialogue History Management Module
Manages conversation context and history for multi-turn conversations
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import uuid


@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ConversationTurn:
    """Represents a single turn (user query + assistant response) in the conversation"""
    user_query: str
    assistant_response: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict = field(default_factory=dict)
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class DialogueHistory:
    """
    Manages conversation history for multi-turn dialogues
    """

    def __init__(self, max_turns: int = 10):
        """
        Initialize dialogue history

        Args:
            max_turns: Maximum number of turns to keep in history
        """
        self.turns: List[ConversationTurn] = []
        self.messages: List[Message] = []
        self.max_turns = max_turns
        self.session_id: str = str(uuid.uuid4())
        self.created_at: datetime = datetime.now()

    def add_turn(
        self,
        user_query: str,
        assistant_response: str,
        metadata: Optional[dict] = None
    ) -> ConversationTurn:
        """Add a new turn to the conversation history"""
        turn = ConversationTurn(
            user_query=user_query,
            assistant_response=assistant_response,
            metadata=metadata or {}
        )

        self.turns.append(turn)
        self.messages.append(Message(role="user", content=user_query))
        self.messages.append(Message(role="assistant", content=assistant_response))

        # Trim if exceeding max turns
        while len(self.turns) > self.max_turns:
            self.turns.pop(0)
            # Remove oldest user and assistant messages
            if len(self.messages) >= 2:
                self.messages = self.messages[2:]

        return turn

    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """Get the most recent n turns"""
        return self.turns[-n:]

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """Get all messages, optionally limited to recent ones"""
        if limit:
            return self.messages[-limit:]
        return self.messages

    def format_history(self, include_system_prompt: bool = False) -> str:
        """
        Format conversation history as a string
        """
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.user_query}")
            lines.append(f"Assistant: {turn.assistant_response}")
        return "\n".join(lines)

    def format_for_llm(self, system_prompt: Optional[str] = None) -> List[dict]:
        """
        Format conversation history for LLM API consumption
        Returns list of message dicts with role and content
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for msg in self.messages:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    def get_context(self) -> str:
        """
        Get conversation context for retrieval
        Returns recent conversation as context string
        """
        if not self.turns:
            return ""

        return self.format_history()

    def clear(self) -> None:
        """Clear all conversation history"""
        self.turns = []
        self.messages = []
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now()

    def get_summary(self) -> dict:
        """Get summary information about the conversation"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "turn_count": len(self.turns),
            "message_count": len(self.messages)
        }

    def find_similar_query(self, query: str, threshold: float = 0.8) -> Optional[ConversationTurn]:
        """
        Find a similar query in history using simple string matching
        Note: For production, use embedding-based similarity
        """
        query_lower = query.lower().strip()

        for turn in reversed(self.turns):
            if turn.user_query.lower().strip() == query_lower:
                return turn

        # Fuzzy match - check if query is contained
        for turn in reversed(self.turns):
            if query_lower in turn.user_query.lower() or turn.user_query.lower() in query_lower:
                return turn

        return None


class DialogueManager:
    """
    Manages multiple conversation sessions
    """

    def __init__(self, max_sessions: int = 100):
        self.sessions: Dict[str, DialogueHistory] = {}
        self.max_sessions = max_sessions

    def create_session(self) -> str:
        """Create a new conversation session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = DialogueHistory()
        return session_id

    def get_session(self, session_id: str) -> Optional[DialogueHistory]:
        """Get a session by ID"""
        return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: Optional[str] = None) -> DialogueHistory:
        """Get existing session or create new one"""
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]

        # Create new session
        new_id = str(uuid.uuid4())
        self.sessions[new_id] = DialogueHistory()

        # Trim old sessions if needed
        while len(self.sessions) > self.max_sessions:
            oldest_id = next(iter(self.sessions))
            del self.sessions[oldest_id]

        return self.sessions[new_id]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def add_to_session(
        self,
        session_id: str,
        user_query: str,
        assistant_response: str,
        metadata: Optional[dict] = None
    ) -> Optional[ConversationTurn]:
        """Add a turn to an existing session"""
        session = self.get_session(session_id)
        if not session:
            return None
        return session.add_turn(user_query, assistant_response, metadata)


if __name__ == "__main__":
    # Example usage
    history = DialogueHistory(max_turns=5)

    # Add some turns
    history.add_turn("你好", "你好！有什么我可以帮助你的吗？")
    history.add_turn("什么是 RAG?", "RAG 是检索增强生成的缩写...")

    # Get formatted history
    print(history.format_history())

    # Format for LLM
    messages = history.format_for_llm()
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:50]}...")
