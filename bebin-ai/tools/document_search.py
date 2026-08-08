from sqlalchemy.orm import Session

from app.db.models import User
from app.services.document_service import search_user_documents
from tools.base import ToolResult


class DocumentSearchTool:
    name = "document_search"
    description = "Search uploaded user documents."

    def __init__(self, db: Session, user: User) -> None:
        self.db = db
        self.user = user

    def run(self, arguments: dict[str, object]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        top_k = int(arguments.get("top_k", 4))
        if not query:
            raise ValueError("document_search requires a query")

        results = search_user_documents(self.db, self.user, query, top_k=top_k)
        content = "\n\n".join(
            f"[{index}] {filename}\n{chunk.text}"
            for index, (chunk, filename) in enumerate(results, start=1)
        )
        return ToolResult(
            name=self.name,
            content=content or "No matching uploaded documents found.",
            metadata={"query": query, "result_count": len(results)},
        )

