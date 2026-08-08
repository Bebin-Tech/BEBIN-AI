from functools import lru_cache
from pathlib import Path
import json

from app.core.config import settings
from app.db.models import Conversation, User
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.document_service import format_rag_context, search_user_documents
from ml.inference.generator import GenerationConfig, TextGenerator
from tools.calculator import CalculatorTool
from tools.document_search import DocumentSearchTool
from tools.planner import format_tool_context, plan_tool_calls
from tools.registry import ToolRegistry
from tools.web_search import WebSearchTool


class ChatServiceError(RuntimeError):
    pass


class ChatService:
    def __init__(self, checkpoint_path: Path, tokenizer_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self._generator: TextGenerator | None = None

    def generate(self, request: ChatRequest, db=None, user: User | None = None) -> ChatResponse:
        rag_context = _rag_context(db, user, request)
        tool_results = _run_tools(db, user, request)
        tool_context = format_tool_context(tool_results)
        generation_prompt = _prompt_with_context(request.message, rag_context, tool_context)
        result = self._get_generator().generate(
            generation_prompt,
            _generation_config_from_request(request),
        )
        conversation_id = None
        if db is not None and user is not None:
            conversation = _get_or_create_conversation(db, user, request)
            _append_message(db, conversation, "user", request.message)
            _append_message(db, conversation, "assistant", result.text)
            db.commit()
            conversation_id = conversation.id
        return ChatResponse(
            conversation_id=conversation_id,
            message=request.message,
            response=result.text,
            rag_context=rag_context or None,
            tool_results=[_tool_result_payload(result) for result in tool_results],
            token_ids=result.token_ids,
            new_token_ids=result.new_token_ids,
            stopped_on_eos=result.stopped_on_eos,
        )

    def stream(self, request: ChatRequest, db=None, user: User | None = None):
        yield "event: start\ndata: {}\n\n"
        generator = self._get_generator()
        config = _generation_config_from_request(request)
        rag_context = _rag_context(db, user, request)
        tool_results = _run_tools(db, user, request)
        tool_context = format_tool_context(tool_results)
        generation_prompt = _prompt_with_context(request.message, rag_context, tool_context)
        prompt_ids = generator.tokenizer.encode(generation_prompt, add_special_tokens=True).ids
        token_ids = list(prompt_ids)
        new_token_ids: list[int] = []
        stopped_on_eos = False

        for token_id, stopped_on_eos in generator.iter_token_ids(generation_prompt, config):
            token_ids.append(token_id)
            new_token_ids.append(token_id)
            text = generator.tokenizer.decode(token_ids, skip_special_tokens=True)
            payload = json.dumps({"token_id": token_id, "text": text})
            yield f"event: token\ndata: {payload}\n\n"

        conversation_id = None
        response_text = generator.tokenizer.decode(token_ids, skip_special_tokens=True)
        if db is not None and user is not None:
            conversation = _get_or_create_conversation(db, user, request)
            _append_message(db, conversation, "user", request.message)
            _append_message(db, conversation, "assistant", response_text)
            db.commit()
            conversation_id = conversation.id

        response = ChatResponse(
            conversation_id=conversation_id,
            message=request.message,
            response=response_text,
            rag_context=rag_context or None,
            tool_results=[_tool_result_payload(result) for result in tool_results],
            token_ids=token_ids,
            new_token_ids=new_token_ids,
            stopped_on_eos=stopped_on_eos,
        )
        payload = response.model_dump_json()
        yield f"event: done\ndata: {payload}\n\n"

    def _get_generator(self) -> TextGenerator:
        if self._generator is not None:
            return self._generator

        checkpoint_path = _resolve_repo_path(self.checkpoint_path)
        tokenizer_path = _resolve_repo_path(self.tokenizer_path)
        missing = [str(path) for path in [checkpoint_path, tokenizer_path] if not path.exists()]
        if missing:
            raise ChatServiceError(
                "Local model artifacts are missing. Train the tokenizer/model first. Missing: "
                + ", ".join(missing)
            )

        self._generator = TextGenerator.from_checkpoint(
            checkpoint_path=checkpoint_path,
            tokenizer_path=tokenizer_path,
        )
        return self._generator


@lru_cache
def get_chat_service() -> ChatService:
    return ChatService(
        checkpoint_path=settings.model_checkpoint_path,
        tokenizer_path=settings.model_tokenizer_path,
    )


def _generation_config_from_request(request: ChatRequest) -> GenerationConfig:
    return GenerationConfig(
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        stop_on_eos=request.stop_on_eos,
        seed=request.seed,
    )


def _resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[4] / path


def _get_or_create_conversation(db, user: User, request: ChatRequest) -> Conversation:
    if request.conversation_id:
        conversation = db.get(Conversation, request.conversation_id)
        if conversation is None or conversation.user_id != user.id:
            raise ValueError("Conversation not found")
        return conversation

    title = request.message[:80] or "New conversation"
    conversation = Conversation(user_id=user.id, title=title)
    db.add(conversation)
    db.flush()
    return conversation


def _append_message(db, conversation: Conversation, role: str, content: str) -> None:
    from app.api.routes.conversations import append_message

    append_message(db, conversation, role, content)


def _rag_context(db, user: User | None, request: ChatRequest) -> str:
    if db is None or user is None or not request.use_rag:
        return ""
    results = search_user_documents(db, user, request.message, top_k=request.rag_top_k)
    return format_rag_context(results)


def _prompt_with_context(message: str, context: str, tool_context: str = "") -> str:
    if not context and not tool_context:
        return message
    sections = []
    if context:
        sections.append("Uploaded document context:\n" + context)
    if tool_context:
        sections.append("Tool results:\n" + tool_context)
    return (
        "Use the following context and tool results if they are relevant.\n\n"
        f"{'\n\n'.join(sections)}\n\n"
        f"User question: {message}"
    )


def _run_tools(db, user: User | None, request: ChatRequest):
    calls = plan_tool_calls(request.message, use_tools=request.use_tools)
    if not calls:
        return []

    tools = [CalculatorTool(), WebSearchTool(enabled=False)]
    if db is not None and user is not None:
        tools.append(DocumentSearchTool(db, user))

    registry = ToolRegistry(tools)
    return registry.run_many(calls)


def _tool_result_payload(result) -> dict[str, object]:
    return {
        "name": result.name,
        "content": result.content,
        "metadata": result.metadata,
    }
