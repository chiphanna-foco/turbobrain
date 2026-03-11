"""FastAPI application entry point for TurboBrain."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import asyncio
import logging
import os
import secrets

from .config import get_settings
from .models.database import (
    init_db, import_knowledge_from_files, import_instant_answers_from_file,
)
from .api.search_api import router as search_router
from .api.admin_api import router as admin_router
from .api.instant_answers_api import router as instant_answers_router
from .api.google_docs_api import router as google_docs_router
from .api.intercom_api import router as intercom_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown."""
    logger.info("Starting TurboBrain...")
    await init_db()
    logger.info("Database initialized")

    import_result = await import_knowledge_from_files()
    await import_instant_answers_from_file(force=True)

    # Generate suggested instant answers for new/updated KB docs
    if import_result and import_result.get("changed_doc_ids"):
        from .analysis.suggestion_generator import generate_suggestions_for_documents
        changed_ids = import_result["changed_doc_ids"]
        logger.info(f"Queueing suggestion generation for {len(changed_ids)} changed KB docs")
        asyncio.create_task(generate_suggestions_for_documents(changed_ids))

    # Run initial Google Docs + Drive folders sync and start periodic refresh
    from .services.google_docs_sync import sync_all_google_docs, sync_all_folders, google_docs_refresh_loop
    gdoc_result = await sync_all_google_docs()
    folder_result = await sync_all_folders()
    all_changed = gdoc_result.get("changed_doc_ids", []) + folder_result.get("changed_doc_ids", [])
    if all_changed:
        from .analysis.suggestion_generator import generate_suggestions_for_documents as gen_sugg
        logger.info(f"Queueing suggestion generation for {len(all_changed)} synced docs")
        asyncio.create_task(gen_sugg(all_changed))
    asyncio.create_task(google_docs_refresh_loop())

    # Run initial Intercom sync and start periodic refresh
    from .services.intercom_sync import sync_all_intercom, intercom_refresh_loop
    await sync_all_intercom()
    asyncio.create_task(intercom_refresh_loop())

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="TurboBrain",
    description="Knowledge Base & ElevenLabs Management",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(search_router)
app.include_router(instant_answers_router)
app.include_router(google_docs_router)
app.include_router(intercom_router)
app.include_router(admin_router)

# Static files for admin dashboard
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Admin authentication
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "turbobrain")


class LoginRequest(BaseModel):
    password: str


@app.post("/api/admin/login")
async def admin_login(body: LoginRequest):
    """Verify admin password."""
    if secrets.compare_digest(body.password, ADMIN_PASSWORD):
        return {"ok": True}
    raise HTTPException(status_code=401, detail="Incorrect password")


@app.get("/admin")
async def admin_page():
    """Serve the admin dashboard."""
    admin_path = os.path.join(static_dir, "admin.html")
    if not os.path.exists(admin_path):
        raise HTTPException(status_code=404, detail="Admin page not found")
    return FileResponse(admin_path, media_type="text/html")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "turbobrain"}


@app.get("/config-check")
async def config_check():
    """Check which integrations are configured."""
    return {
        "anthropic": bool(settings.anthropic_api_key),
        "elevenlabs": bool(settings.elevenlabs_api_key),
        "elevenlabs_agent": bool(settings.elevenlabs_agent_id),
        "google_drive": bool(settings.google_drive_api),
        "database": "postgresql" if "postgresql" in settings.effective_database_url else "sqlite",
    }
