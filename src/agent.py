from typing import Callable

from .store import EmbeddingStore

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý chuyên về văn bản pháp luật Việt Nam.

Quy tắc trả lời:
1. Chỉ trả lời dựa trên các điều khoản được cung cấp trong phần "Ngữ cảnh" — không suy diễn ngoài văn bản.
2. Khi trả lời, luôn trích dẫn nguồn cụ thể theo dạng (Điều X, Khoản Y — [số hiệu nghị định]) ở cuối câu liên quan.
3. Nếu các điều khoản được cung cấp KHÔNG chứa đủ thông tin để trả lời câu hỏi, hãy trả lời: "Tôi không tìm thấy thông tin liên quan trong các văn bản được cung cấp. Vui lòng tham khảo trực tiếp văn bản pháp luật hoặc tư vấn chuyên gia."
4. Trả lời ngắn gọn, rõ ràng bằng tiếng Việt."""


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    RAG pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build system prompt + user message with cited context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        chunks = self.store.search(question, top_k=top_k)
        context_blocks = "\n\n".join(
            f"[{i+1}] {c['content']}" for i, c in enumerate(chunks)
        )
        # Pass system + user as a single string with clear delimiters
        # so llm_fn implementations can parse if needed, or use as-is
        prompt = (
            f"<system>\n{SYSTEM_PROMPT}\n</system>\n\n"
            f"<user>\n"
            f"Ngữ cảnh từ văn bản pháp luật:\n{context_blocks}\n\n"
            f"Câu hỏi: {question}\n"
            f"</user>"
        )
        return self.llm_fn(prompt)
