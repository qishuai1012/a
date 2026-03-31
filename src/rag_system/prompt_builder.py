"""
RAG System - Prompt Builder Module
Constructs prompts for the LLM with retrieved context
"""

from typing import List, Optional, Dict
from dataclasses import dataclass


@dataclass
class PromptContext:
    """Represents context for prompt construction"""
    query: str
    retrieved_documents: List[str]
    conversation_history: List[dict]
    system_prompt: Optional[str] = None


class PromptBuilder:
    """
    Builds prompts for the LLM with retrieved context
    """

    DEFAULT_SYSTEM_PROMPT = """你是一个智能助手，基于检索到的信息为用户提供准确的答案。

回答要求：
1. 优先基于提供的上下文信息回答问题
2. 如果上下文信息不足，明确说明
3. 不要编造信息
4. 回答简洁明了
5. 如果问题与上下文无关，礼貌地引导用户"""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        template: Optional[str] = None
    ):
        """
        Initialize prompt builder

        Args:
            system_prompt: System prompt for the LLM
            template: Custom template for the prompt
        """
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.template = template or self._default_template()

    def _default_template(self) -> str:
        """Default prompt template"""
        return """<context>
{context}
</context>

<conversation_history>
{history}
</conversation_history>

<user_question>
{question}
</user_question>

请根据上述上下文和对话历史，回答用户的问题。"""

    def build(
        self,
        query: str,
        documents: List[str],
        conversation_history: Optional[List[dict]] = None,
        system_prompt: Optional[str] = None
    ) -> List[dict]:
        """
        Build a prompt for the LLM

        Args:
            query: User's query
            documents: List of retrieved documents
            conversation_history: List of previous messages
            system_prompt: Optional custom system prompt

        Returns:
            List of message dicts for LLM API
        """
        # Build context from documents
        context = self._build_context(documents)

        # Build conversation history
        history = self._build_history(conversation_history or [])

        # Build user message
        user_content = self.template.format(
            context=context,
            history=history,
            question=query
        )

        # Construct messages
        messages = []

        # System prompt
        messages.append({
            "role": "system",
            "content": system_prompt or self.system_prompt
        })

        # Conversation history
        if conversation_history:
            messages.extend(conversation_history)

        # User message
        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages

    def _build_context(self, documents: List[str]) -> str:
        """Build context string from documents"""
        if not documents:
            return "未检索到相关文档。"

        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"[相关片段 {i}]\n{doc}")

        return "\n\n".join(context_parts)

    def _build_history(self, history: List[dict]) -> str:
        """Build history string from conversation messages"""
        if not history:
            return "无历史对话。"

        history_parts = []
        for msg in history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            role_name = "用户" if role == "user" else "助手"
            history_parts.append(f"{role_name}: {content}")

        return "\n".join(history_parts)

    def build_simple(
        self,
        query: str,
        documents: List[str]
    ) -> str:
        """
        Build a simple prompt without conversation history

        Args:
            query: User's query
            documents: List of retrieved documents

        Returns:
            Simple prompt string
        """
        context = self._build_context(documents)

        return f"""基于以下信息回答问题：

{context}

问题：{query}

回答："""


class StreamingPromptBuilder(PromptBuilder):
    """
    Prompt builder optimized for streaming responses
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build_for_streaming(
        self,
        query: str,
        documents: List[str],
        conversation_history: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Build prompt for streaming API

        Same as build() but ensures compatibility with streaming endpoints
        """
        return self.build(query, documents, conversation_history)


class MinimalPromptBuilder:
    """
    Minimal prompt builder for simple use cases
    """

    def build(self, query: str, context: str) -> str:
        """Build minimal prompt"""
        return f"""上下文：{context}

问题：{query}

回答："""


if __name__ == "__main__":
    # Example usage
    builder = PromptBuilder()

    documents = [
        "RAG 是检索增强生成的缩写",
        "RAG 结合了检索和生成的优势"
    ]

    messages = builder.build(
        query="什么是 RAG？",
        documents=documents,
        conversation_history=[]
    )

    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:100]}...")
