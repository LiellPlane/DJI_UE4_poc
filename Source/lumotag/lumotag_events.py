from pydantic import BaseModel, Field, field_validator
import time

class ReqKillScreenResponse(BaseModel):
    display_name_tagger: str
    image_datas: list[str] = Field(..., description="Base64 encoded JPEG image data for HTTP transmission")
    event_type: str = Field(default_factory=lambda: "ReqKillScreenResponse", description="Event type identifier")

class UploadRequest(BaseModel):
    """Pydantic model for validating upload request data"""
    image_id: str = Field(..., description="Unique identifier for the image")
    image_data: str = Field(..., description="Base64 encoded JPEG image data for HTTP transmission")
    event_type: str = Field(default_factory=lambda: "UploadRequest", description="Event type identifier")

class PlayerStatus(BaseModel):
    health: int
    ammo: int
    tag_id: str
    display_name: str
    is_connected: bool
    last_active: int
    event_type: str = Field(default_factory=lambda: "PlayerStatus", description="Event type identifier")


class GameStatus(BaseModel):
    players: dict[str, PlayerStatus]  # Key is player's tag_id
    event_type: str = Field(default_factory=lambda: "GameStatus", description="Event type identifier")


class PlayerTagged(BaseModel):
    tag_id: str
    image_ids: list[str] = Field(..., description="Unique identifier for the image(s) captured during tag")
    event_type: str = Field(default_factory=lambda: "PlayerTagged", description="Event type identifier")
    
    @field_validator('tag_id', mode='before')
    @classmethod
    def convert_tag_id_to_str(cls, v):
        return str(v)
