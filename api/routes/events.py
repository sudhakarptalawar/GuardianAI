"""
GuardianAI — Events API Route
GET  /events          → list all events (with optional filters)
GET  /events/{id}     → get single event
GET  /events/summary  → severity summary counts
DELETE /events        → clear all events
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional

from core.event_manager import EventManager
from core.severity       import Severity
from api.models.schemas  import EventSchema, EventSummarySchema

router = APIRouter(prefix="/events", tags=["Events"])

# Shared event manager instance (injected by main app)
_event_manager: EventManager | None = None


def set_event_manager(em: EventManager):
    global _event_manager
    _event_manager = em


def get_em() -> EventManager:
    if _event_manager is None:
        raise HTTPException(status_code=503, detail="Event manager not initialized.")
    return _event_manager


@router.get("/summary", response_model=EventSummarySchema)
def get_summary():
    """Return total event count grouped by severity."""
    return get_em().summary()


@router.get("/", response_model=List[EventSchema])
def list_events(
    severity:  Optional[str] = Query(None, description="Filter by severity: LOW|MEDIUM|HIGH|CRITICAL"),
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    limit:     int           = Query(100, ge=1, le=1000, description="Max results to return"),
):
    """Return all events, optionally filtered by severity or camera."""
    em = get_em()

    if severity:
        try:
            sev    = Severity(severity.upper())
            events = em.get_by_severity(sev)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid severity: {severity}")
    elif camera_id:
        events = em.get_by_camera(camera_id)
    else:
        events = em.get_all()

    # Return most recent first, up to limit
    events = list(reversed(events))[:limit]
    return [EventSchema(**e.__dict__) for e in events]


@router.get("/{event_id}", response_model=EventSchema)
def get_event(event_id: str):
    """Return a single event by ID."""
    events = get_em().get_all()
    for e in events:
        if e.id == event_id:
            return EventSchema(**e.__dict__)
    raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")


@router.delete("/")
def clear_events():
    """Clear all events from memory and disk."""
    em = get_em()
    em.events.clear()
    em._save()
    return {"success": True, "message": "All events cleared."}
