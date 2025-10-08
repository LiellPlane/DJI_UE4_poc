from pydantic import BaseModel, Field, field_validator
from string import Template
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
Template("Study harder $player_name."),
Template("Blade found $player_name."),
Template("$player_name got TOUCHED."),
Template("$player_name Tasted the lightning."),
Template("$player_name SLAPPED by fate."),
Template("The blade whispers $player_name"),
Template("Timing beats $player_name."),
Template("TAGGED. Reflect now $player_name."),
Template("Swift justice delivered to $player_name"),
Template("$player_name blinked. Mistake."),
Template("Caught $player_name slipping."),
Template("The blade remembers $player_name"),
Template("$player_name BONKED by destiny."),
Template("Discipline > $player_name."),
Template("Et tu, $player_name?"),
Template("Friendship failed $player_name."),
Template("Skill issue, $player_name."),
Template("$player_name: Simply outplayed."),
Template("The council remembers, $player_name."),
Template("Winter came for $player_name."),
Template("Not today, $player_name."),
Template("Tagged, $player_name was."),
Template("Much to learn, $player_name has."),
Template("Ready, $player_name was not."),
Template("Like, bye $player_name."),
Template("$player_name? So last season."),
Template("Ugh, $player_name. That's tragic."),
Template("$player_name: Zero stars. Next."),
Template("Too soon, $player_name."),
Template("$player_name granny-shifted."),
Template("Family > $player_name."),
Template("$player_name will NOT see Rome."),
Template("$player_name couldn't handle the NOS."),
Template("Are you entertained, $player_name?"),
Template("$player_name will not echo in eternity"),
Template("$player_name: Zero sass detected."),
Template("$player_name: not ready for this jelly"),
Template("$player_name: Inadequate pythons."),
Template("$player_name: Title shot denied."),
Template("Behave yourself, $player_name."),
Template("$player_name: To bed without supper."),
Template("$player_name: Most unseemly conduct."),
Template("For shame, $player_name."),
Template("$player_name: No pudding tonight."),
Template("Mind your manners, $player_name."),
Template("Cease this nonsense, $player_name."),
Template("$player_name: Latin conjugations, now."),
Template("Father shall hear of this, $player_name."),
Template("$player_name: Recite scripture. Again."),
Template("$player_name: Etiquette class extended."),
Template("$player_name: Needlework until bedtime."),
Template("$player_name: My affections lie elsewhere."),
Template("Unhand me, $player_name."),
Template("$player_name: You presume too much."),
Template("I think not, $player_name."),
Template("$player_name: Our stations differ greatly."),
Template("Impossible, dear $player_name."),
Template("$player_name: My heart belongs to another."),
Template("Declined, $player_name. Most firmly."),
Template("$player_name: We are not suited."),
Template("$player_name: I could never love thee."),
Template("$player_name: That's life, pal."),
Template("Wise up, $player_name."),
Template("$player_name: My guardian disapproves."),
Template("Not you, $player_name. Never you."),
Template("$player_name: I am already promised."),
Template("Heavens no, $player_name."),
Template("$player_name: My reputation, sir!"),
Template("I cannot, $player_name. I shan't."),
Template("$player_name: You're too late, darling."),
Template("$player_name: What would people say?"),
Template("When doves cry for $player_name."),
]

bullied_chat: list = [
    Template("$player_name departed with predujice at $health HP"),
    Template("Stop! Stop! $player_name's already dead at $health HP!"),
    Template("Unnecessary roughness: $player_name at $health HP"),
    Template("Brutality! $player_name fell long ago before $health HP!"),
    Template("$player_name at $health HP. Have you no honor?"),
    Template("Unseemly conduct upon $player_name at $health HP."),
    Template("$player_name transcended at $health HP"),
    Template("$player_name at $health HP sends its regards."),
    Template("$player_name RIP mutiplier at $health HP"),
    Template("$player_name left the chat at $health HP"),
    Template("$player_name studied the blade to $health HP"),
    Template("$player_name war-crimed to $health HP"),
    Template("$player_name at $health HP seeking their next life"),
    Template("$player_name transcends at $health HP. Let them go."),
    Template("$player_name at $health HP enters the void."),
    Template("$player_name at $health HP reborn soon."),
    Template("Karma delivered to $player_name at $health H."),
    Template("Samsara claims $player_name at $health HP"),
    Template("$player_name at $health HP achieving enlightenment despite you"),
    Template("The ancestors call $player_name at $health HP"),
    Template("Reincarnation queue: $player_name at $health HP"),
    Template("Metaphysical overkill on $player_name at $health HP"),
]
