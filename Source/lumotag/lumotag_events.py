from pydantic import BaseModel, Field
import time


class UploadRequest(BaseModel):
    """Pydantic model for validating upload request data"""
    image_id: str = Field(..., description="Unique identifier for the image")
    timestamp: float = Field(default_factory=time.time, description="Unix timestamp when upload was initiated")


class PlayerStatus(BaseModel):
    health: int
    ammo: int
    tag_id: str
    display_name: str


class GameUpdate(BaseModel):
    players: list[PlayerStatus]


class PlayerTagged(BaseModel):
    tag_id: str
    image_ids: list[str] = Field(..., description="Unique identifier for the image(s) captured during tag")
