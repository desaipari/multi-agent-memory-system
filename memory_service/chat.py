"""
Chat and file upload endpoints.
These receive input from the React dashboard and route
it through the LangGraph agent pipeline.

Person B owns this file.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import shutil

router = APIRouter()

# Import agent pipeline functions
# These are in the agent_pipeline folder
# We add it to path so FastAPI can import them
AGENT_PIPELINE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "agent_pipeline"
)
sys.path.insert(0, AGENT_PIPELINE_PATH)


class ChatRequest(BaseModel):
    message: str
    agent_role: str = "intake"  # intake, billing, coordinator
    source_file: Optional[str] = "manual_input"


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Receive a text message from the dashboard and route it
    through the appropriate LangGraph agent.

    agent_role determines which agent processes the message:
    - intake: Intake Agent (reads ticket system data)
    - billing: Billing/Ops Agent (reads field reports)
    - coordinator: Coordinator Agent (resolves conflicts)
    """
    try:
        from graph import run_intake, run_billing
        from agents.coordinator_agent import process_flagged_conflicts

        if request.agent_role == "billing":
            result = run_billing(
                request.message,
                request.source_file
            )
        elif request.agent_role == "coordinator":
            result = process_flagged_conflicts()
            return {
                "success": True,
                "agent_role": "coordinator",
                "conflicts_resolved": result.get("resolved", 0),
                "resolutions": result.get("resolutions", []),
                "message": (
                    f"Coordinator resolved "
                    f"{result.get('resolved', 0)} conflicts"
                )
            }
        else:
            result = run_intake(
                request.message,
                request.source_file
            )

        return {
            "success": result["status"] not in [
                "intake_failed", "billing_failed"
            ],
            "status": result["status"],
            "agent_role": request.agent_role,
            "extracted_fact": result.get("extracted_fact"),
            "incident_id": result.get("incident_id"),
            "contradiction_detected": result.get(
                "contradiction_detected", False
            ),
            "contradiction_details": result.get(
                "contradiction_details"
            ),
            "recommendation": result.get("recommendation"),
            "error": result.get("error")
        }

    except ImportError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Agent pipeline not found: {e}. "
                f"Make sure agent_pipeline folder is at "
                f"{AGENT_PIPELINE_PATH}"
            )
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent processing error: {str(e)}"
        )


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    agent_role: str = Form(default="intake")
):
    """
    Receive a CSV or PDF file from the dashboard.
    Processes it through the file_processor and returns results.
    """
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    # Check file type
    allowed_types = [".csv", ".pdf"]
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Use: {allowed_types}"
        )

    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=file_ext,
            prefix="upload_"
        ) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {e}"
        )

    # Process the file
    try:
        from file_processor import process_csv, process_pdf

        if file_ext == ".csv":
            result = process_csv(tmp_path, agent_role)
        else:
            result = process_pdf(tmp_path, agent_role)

        return {
            "success": result.get("success", False),
            "filename": file.filename,
            "agent_role": agent_role,
            "rows_processed": result.get("rows_processed", 0),
            "sentences_processed": result.get("sentences_processed", 0),
            "contradictions_found": result.get("contradictions_found", 0),
            "errors": result.get("errors", 0),
            "message": (
                f"Processed {file.filename}: "
                f"{result.get('rows_processed') or result.get('sentences_processed', 0)} "
                f"facts written, "
                f"{result.get('contradictions_found', 0)} contradictions found"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"File processing error: {str(e)}"
        )
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except Exception:
            pass