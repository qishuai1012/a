"""
Dialogue History Management Module
Manages conversation context and history for multi-turn conversations
"""

from typing import List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
import uuid

#单条消息
@dataclass
class Message:
    """Represents a single message in the conversation"""
    role: str  # 角色："user"（用户）或 "assistant"（助手）
    content: str  # 消息的具体内容
    timestamp: datetime = field(default_factory=datetime.now)  # 发送时间：自动记录当前时间
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 唯一ID：自动生成一个随机字符串

#一轮对话
@dataclass
class ConversationTurn:
    """Represents a single turn (user query + assistant response) in the conversation"""
    user_query: str  # 用户问了啥
    assistant_response: str  # AI 回了啥
    timestamp: datetime = field(default_factory=datetime.now)  # 时间
    metadata: dict = field(default_factory=dict)  # 元数据：比如可以存“这次回答耗时多少”、“用了什么模型”
    turn_id: str = field(default_factory=lambda: str(uuid.uuid4()))  # 这一轮对话的ID

#单会话管理
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
        self.turns: List[ConversationTurn] = []  # 存放所有的“轮次”
        self.messages: List[Message] = []        # 存放所有的“单条消息”（扁平化列表）
        self.max_turns = max_turns               # 最大记忆长度：默认只记最近的 10 轮
        self.session_id: str = str(uuid.uuid4()) # 这次聊天的唯一身份证号
        self.created_at: datetime = datetime.now() # 聊天开始的时间

    #添加对话 (add_turn)
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

        # 2. 存入列表
        self.turns.append(turn)  # 存入轮次列表
        # 同时也存入扁平的消息列表（先存用户，再存助手）
        self.messages.append(Message(role="user", content=user_query))
        self.messages.append(Message(role="assistant", content=assistant_response))

        # 3. 记忆修剪（核心！）
        # 如果轮次超过了最大限制（比如 10 轮）
        while len(self.turns) > self.max_turns:
            self.turns.pop(0)  # 删掉最早的轮次
            # 删掉最早的两条消息（一问一答）
            if len(self.messages) >= 2:
                self.messages = self.messages[2:]

        return turn

    #获取数据 (get_recent_turns, get_messages)
    def get_recent_turns(self, n: int = 5) -> List[ConversationTurn]:
        """获取最近的 n 轮对话"""
        return self.turns[-n:]  # 切片：取最后 n 个

    def get_messages(self, limit: Optional[int] = None) -> List[Message]:
        """获取消息列表"""
        if limit:
            return self.messages[-limit:]  # 如果有限制，取最后几个
        return self.messages  # 否则全给

    #格式化输出 (format_history, format_for_llm)
    def format_history(self, include_system_prompt: bool = False) -> str:
        """
        把历史记录变成一段纯文本
        比如：
        User: 你好
        Assistant: 你好
        """
        lines = []
        for turn in self.turns:
            lines.append(f"User: {turn.user_query}")
            lines.append(f"Assistant: {turn.assistant_response}")
        return "\n".join(lines)

    def format_for_llm(self, system_prompt: Optional[str] = None) -> List[dict]:
        """
        把历史记录变成 LLM API 需要的格式（JSON 列表）
        这是调用 ChatGPT/Claude 等接口时必须用的格式
        """
        messages = []

        # 1. 先加系统提示词（如果有的话）
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # 2. 再加历史消息
        for msg in self.messages:
            messages.append({"role": msg.role, "content": msg.content})

        return messages

    def get_context(self) -> str:
        """
        Get conversation context for retrieval
        Returns recent conversation as context string
        """
        """获取对话上下文（用于检索或提示）"""

        if not self.turns:
            return ""

        return self.format_history()

    #清空记忆
    def clear(self) -> None:
        """Clear all conversation history"""
        self.turns = []
        self.messages = []
        self.session_id = str(uuid.uuid4())   # 换个新ID，相当于开启新聊天
        self.created_at = datetime.now()

    #获取摘要
    def get_summary(self) -> dict:
        """Get summary information about the conversation"""
        """获取这次聊天的统计信息"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "turn_count": len(self.turns),
            "message_count": len(self.messages)
        }

    #查找相似问题
    def find_similar_query(self, query: str, threshold: float = 0.8) -> Optional[ConversationTurn]:
        """
        在历史记录里找类似的问题（简单的字符串匹配）
        注意：代码注释里说了，生产环境最好用向量相似度，这里只是简单的文本匹配
        """
        query_lower = query.lower().strip()  # 转小写、去空格

        # 1. 精确匹配：倒着找（从最近的开始找）
        for turn in reversed(self.turns):
            if turn.user_query.lower().strip() == query_lower:
                return turn  # 找到完全一样的

        # 2. 包含匹配：如果问题里包含了以前的词，或者以前的问题包含现在的词
        for turn in reversed(self.turns):
            if query_lower in turn.user_query.lower() or turn.user_query.lower() in query_lower:
                return turn

        return None  # 没找到

#多会话管理
class DialogueManager:
    """
    Manages multiple conversation sessions
    """

    def __init__(self, max_sessions: int = 100):
        self.sessions: Dict[str, DialogueHistory] = {}  # 存放所有的会话，key是session_id
        self.max_sessions = max_sessions  # 最多同时支持多少个会话

    def create_session(self) -> str:
        """创建一个新会话"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = DialogueHistory()  # 新建一个历史记录对象
        return session_id  # 返回ID给用户，用户下次带着这个ID来

    #获取会话 (get_session, get_or_create_session)
    def get_session(self, session_id: str) -> Optional[DialogueHistory]:
        """根据ID获取会话"""
        return self.sessions.get(session_id)

    def get_or_create_session(self, session_id: Optional[str] = None) -> DialogueHistory:
        """获取已有会话，如果没有就创建一个新的"""
        # 如果给了ID且存在，就返回现有的
        if session_id and session_id in self.sessions:
            return self.sessions[session_id]

        # 否则创建新的
        new_id = str(uuid.uuid4())
        self.sessions[new_id] = DialogueHistory()

        # 如果会话太多，删掉最老的（字典是有序的）
        while len(self.sessions) > self.max_sessions:
            oldest_id = next(iter(self.sessions))
            del self.sessions[oldest_id]

        return self.sessions[new_id]

    #删除会话
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        """删除指定会话"""
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

#对话记忆系统
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
