from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.deps import DbSession, get_current_user
from app.db.models import Resume, User
from app.schemas.resume import ResumeChatRequest, ResumeChatResponse, ResumeResponse, ResumeSearchResult
from app.services.resume_rag_service import (
    ResumeRagError,
    answer_resume_question,
    ingest_resume_pdfs,
    list_user_resumes,
    search_resumes,
)

router = APIRouter(prefix="/resumes", tags=["resume-rag"])


@router.post("/bulk", response_model=list[ResumeResponse])
def upload_resume_bulk(
    db: DbSession,
    user: User = Depends(get_current_user),
    files: list[UploadFile] = File(...),
) -> list[ResumeResponse]:
    try:
        resumes = ingest_resume_pdfs(db, user, files)
    except ResumeRagError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return [_resume_response(resume) for resume in resumes]


@router.get("", response_model=list[ResumeResponse])
def list_resumes(
    db: DbSession,
    user: User = Depends(get_current_user),
) -> list[ResumeResponse]:
    return [_resume_response(resume) for resume in list_user_resumes(db, user)]


@router.get("/search", response_model=list[ResumeSearchResult])
def search_resume_index(
    db: DbSession,
    user: User = Depends(get_current_user),
    query: str = Query(..., min_length=1),
    candidate_name: str | None = Query(default=None),
    top_k: int = Query(default=5, ge=1, le=20),
) -> list[ResumeSearchResult]:
    return [
        _search_result(chunk, resume)
        for chunk, resume in search_resumes(db, user, query, candidate_name=candidate_name, top_k=top_k)
    ]


@router.post("/chat", response_model=ResumeChatResponse)
def resume_chat(
    request: ResumeChatRequest,
    db: DbSession,
    user: User = Depends(get_current_user),
) -> ResumeChatResponse:
    answer, matches = answer_resume_question(
        db,
        user,
        request.question,
        candidate_name=request.candidate_name,
        top_k=request.top_k,
    )
    return ResumeChatResponse(
        answer=answer,
        matches=[_search_result(chunk, resume) for chunk, resume in matches],
    )


def _resume_response(resume: Resume) -> ResumeResponse:
    return ResumeResponse(
        id=resume.id,
        filename=resume.filename,
        candidate_name=resume.candidate_name,
        email=resume.email,
        phone=resume.phone,
        skills=resume.skills,
        chunk_count=len(resume.chunks),
        created_at=resume.created_at,
    )


def _search_result(chunk, resume: Resume) -> ResumeSearchResult:
    return ResumeSearchResult(
        resume_id=resume.id,
        chunk_id=chunk.chunk_id,
        candidate_name=resume.candidate_name,
        filename=resume.filename,
        text=chunk.text,
        score=chunk.score,
    )
