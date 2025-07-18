use serde::{Deserialize, Serialize};



#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Player {
    pub id: String,
    pub name: String,
    pub health: u8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameStatus {
    pub players: Vec<Player>,
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub id: String,
    pub name: String,
    pub health: u8
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagImage {
    pub id: String,
    pub image_data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GameMessage {
    pub timestamp: u64,
    pub payload: Option<GameMessagePayload>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GameMessagePayload {
    Connection(Connection),
    Tag(Tag),
    TagImage(TagImage),
    GameStatus(GameStatus),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    pub player_name: String,
    pub model: String,
}


