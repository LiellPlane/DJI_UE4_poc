from pydantic import BaseModel, Field, field_validator
import time
import uuid

class ReqKillScreenResponse(BaseModel):
    display_name_tagger: str
    image_datas: list[str] = Field(..., description="Base64 encoded JPEG image data for HTTP transmission")
    event_type: str = Field(default_factory=lambda: "ReqKillScreenResponse", description="Event type identifier")

class ReqKillScreen(BaseModel):
    event_type: str = Field(default_factory=lambda: "ReqKillScreen", description="Event type identifier")

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
    isEliminated: bool
    event_type: str = Field(default_factory=lambda: "PlayerStatus", description="Event type identifier")


class GameStatus(BaseModel):
    players: dict[str, PlayerStatus]  # Key is player's device_id
    event_type: str = Field(default_factory=lambda: "GameStatus", description="Event type identifier")


class PlayerTagged(BaseModel):
    tag_id: str
    image_ids: list[str] = Field(..., description="Unique identifier for the image(s) captured during tag")
    tag_uuid: str = Field(default_factory=lambda: uuid.uuid4().hex, description="Unique identifier for this tag event (32 char hex, no hyphens)")
    event_type: str = Field(default_factory=lambda: "PlayerTagged", description="Event type identifier")
    
    @field_validator('tag_id', mode='before')
    @classmethod
    def convert_tag_id_to_str(cls, v):
        return str(v)

eliminated_chat: list = [
    "He who hesitates is STRAIGHT UP COOKED, my guy",
    "Swift as the wind, deadly as fire - YOU? COOKED, BOI",
    "The warrior who masters timing masters all - you got NO TIMING, got SENT",
    "Strike when they blink - they NEVER blinked and you got FOLDED",
    "Know thyself - clearly you DON'T, got SMOKED",
    "The blade flows like water - you DROWNED in it, homie",
    "A warrior's greatest weapon is patience - yours RAN OUT, got CLAPPED",
    "Victory favors the prepared - you came UNPREPARED, got WRECKED",
    "Move like shadow, strike like thunder - they were BOTH and you got DUSTED",
    "The untrained warrior swings wildly - that's YOU, you just got DEMOLISHED",
    "Discipline defeats chaos every time - you brought CHAOS, got SCHOOLED",
    "One who knows when to fight wins - you DIDN'T KNOW, got absolutely MERKED",
    "The superior warrior attacks strategy - they attacked YOU, straight BODIED",
    "Master the fundamentals before the blade - you mastered NOTHING, got YEETED",
    "Timing separates victory from defeat - your timing? TRASH, got OBLITERATED",
    "The patient hunter always eats - you? STARVING and COOKED, kid",
    "Speed without precision is just noise - you were LOUD and got SENT TO OBLIVION",
    "The warrior moves when the enemy sleeps - you SLEPT and got DUMPSTERED",
    "Control the engagement, control the outcome - you controlled NOTHING, got REKT",
    "The wise deflect attacks - you ABSORBED everything, got absolutely FOLDED",
]

tagged_chat: list = [
"BOOM. Study harder {player_name}.",
"Blade found {player_name}.",
"{player_name} got TOUCHED.",
"{player_name} Tasted the lightning.",
"Shadow says hi.",
"{player_name} SLAPPED by fate.",
"The blade whispers {player_name}",
"Timing beats {player_name}.",
"TAGGED. Reflect now {player_name}.",
"Swift justice delivered to {player_name}",
"{player_name} blinked. Mistake.",
"Caught {player_name} slipping.",
"The blade remembers {player_name}",
"{player_name} BONKED by destiny.",
"Discipline > {player_name}.",
"Et tu, {player_name}?",
"Friendship failed {player_name}.",
"Skill issue, {player_name}.",
"{player_name}: Simply outplayed.",
"The council remembers, {player_name}.",
"Winter came for {player_name}.",
"Not today, {player_name}.",
"Tagged, {player_name} was.",
"Much to learn, {player_name} has.",
"Ready, {player_name} was not.",
"Like, bye {player_name}.",
"{player_name}? So last season.",
"Ugh, {player_name}. That's tragic.",
"{player_name}: Zero stars. Next.",
"Too soon, {player_name}.",
"{player_name} granny-shifted.",
"Ride or die? {player_name} died.",
"Family > {player_name}.",
"{player_name} will NOT see Rome.",
"{player_name} couldn't handle the NOS.",
"Are you entertained, {player_name}?",
]