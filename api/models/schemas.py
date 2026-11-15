"""
GuardianAI — Pydantic Schemas
Shared request/response models for the API.
"""

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class EventSchema(BaseModel):
    id:          str
    timestamp:   str
    camera_id:   str
    category:    str
    severity:    str
    label:       str
    description: str
    confidence:  float
    clip_path:   Optional[str] = None


class EventSummarySchema(BaseModel):
    total_events: int
    by_severity:  dict[str, int]


class CameraStatusSchema(BaseModel):
    camera_id:   str
    name:        str
    connected:   bool
    fps:         float
    frame_count: int
    last_error:  str


class AlertConfigSchema(BaseModel):
    enabled:      bool          = True
    min_severity: str           = "HIGH"       # LOW | MEDIUM | HIGH | CRITICAL
    email:        Optional[str] = None
    phone:        Optional[str] = None


class AlertSchema(BaseModel):
    id:        str
    timestamp: str
    event_id:  str
    channel:   str                             # email | sms | websocket
    message:   str
    delivered: bool


class APIResponse(BaseModel):
    success: bool
    message: str
    data:    Optional[dict] = None
