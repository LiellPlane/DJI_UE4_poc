from pydantic import BaseModel, Field
import time


class UploadRequest(BaseModel):
    """Pydantic model for validating upload request data"""
    image_id: str = Field(..., description="Unique identifier for the image")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp when upload was initiated")
    image_data: str = Field(..., description="Base64 encoded JPEG image data for HTTP transmission")
    event_type: str = Field(default_factory=lambda: "UploadRequest", description="Event type identifier")


class PlayerStatus(BaseModel):
    health: int
    ammo: int
    tag_id: str
    display_name: str
    event_type: str = Field(default_factory=lambda: "PlayerStatus", description="Event type identifier")


class GameUpdate(BaseModel):
    players: dict[str, PlayerStatus]  # Key is player's tag_id
    event_type: str = Field(default_factory=lambda: "GameUpdate", description="Event type identifier")


class PlayerTagged(BaseModel):
    tag_id: str
    image_ids: list[str] = Field(..., description="Unique identifier for the image(s) captured during tag")
    event_type: str = Field(default_factory=lambda: "PlayerTagged", description="Event type identifier")
