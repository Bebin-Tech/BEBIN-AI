from functools import lru_cache
from pathlib import Path
import json

from app.core.config import settings
from app.schemas.chat import ChatRequest, ChatResponse
from ml.inference.generator import GenerationConfig, TextGenerator


class ChatServiceError(RuntimeError):
    pass


class ChatService:
    def __init__(self, checkpoint_path: Path, tokenizer_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self._generator: TextGenerator | None = None

    def generate(self, request: ChatRequest) -> ChatResponse:
        result = self._get_generator().generate(
            request.message,
            _generation_config_from_request(request),
        )
        return ChatResponse(
            message=request.message,
            response=result.text,
            token_ids=result.token_ids,
            new_token_ids=result.new_token_ids,
            stopped_on_eos=result.stopped_on_eos,
        )

    def stream(self, request: ChatRequest):
        yield "event: start\ndata: {}\n\n"
        generator = self._get_generator()
        config = _generation_config_from_request(request)
        prompt_ids = generator.tokenizer.encode(request.message, add_special_tokens=True).ids
        token_ids = list(prompt_ids)
        new_token_ids: list[int] = []
        stopped_on_eos = False

        for token_id, stopped_on_eos in generator.iter_token_ids(request.message, config):
            token_ids.append(token_id)
            new_token_ids.append(token_id)
            text = generator.tokenizer.decode(token_ids, skip_special_tokens=True)
            payload = json.dumps({"token_id": token_id, "text": text})
            yield f"event: token\ndata: {payload}\n\n"

        response = ChatResponse(
            message=request.message,
            response=generator.tokenizer.decode(token_ids, skip_special_tokens=True),
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
