from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.deps import DbSession, get_current_user
from app.db.models import Document, User
from app.schemas.document import DocumentResponse, SearchResult
from app.services.document_service import DocumentServiceError, ingest_document, search_user_documents

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse)
def upload_document(
    db: DbSession,
    user: User = Depends(get_current_user),
    file: UploadFile = File(...),
) -> DocumentResponse:
    try:
        document = ingest_document(db, user, file)
    except DocumentServiceError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _document_response(document)


@router.get("", response_model=list[DocumentResponse])
def list_documents(
    db: DbSession,
    user: User = Depends(get_current_user),
) -> list[DocumentResponse]:
    documents = db.query(Document).filter(Document.user_id == user.id).order_by(Document.created_at.desc()).all()
    return [_document_response(document) for document in documents]


@router.get("/search", response_model=list[SearchResult])
def search_documents(
    db: DbSession,
    user: User = Depends(get_current_user),
    query: str = Query(..., min_length=1),
    top_k: int = Query(default=4, ge=1, le=20),
) -> list[SearchResult]:
    results = search_user_documents(db, user, query, top_k=top_k)
    return [
        SearchResult(
            document_id=chunk.document_id,
            chunk_id=chunk.chunk_id,
            filename=filename,
            text=chunk.text,
            score=chunk.score,
        )
        for chunk, filename in results
    ]


def _document_response(document: Document) -> DocumentResponse:
    return DocumentResponse(
        id=document.id,
        filename=document.filename,
        content_type=document.content_type,
        chunk_count=len(document.chunks),
        created_at=document.created_at,
    )

