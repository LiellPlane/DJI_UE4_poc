use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tokio::sync::broadcast;

#[derive(Clone, Serialize, Deserialize)]
struct GameStatus {
    tagged_players: Vec<String>,
    active_players: Vec<String>,
    // Add any other game state info you want to share
}

#[derive(Clone, Serialize, Deserialize)]
enum GameMessage {
    Tag { tagger_id: String, tagged_id: String },
    StateUpdate(GameStatus),
    Error { message: String },
}

struct Player {
    id: String,
    is_tagged: bool,
    ws_sender: broadcast::Sender<GameMessage>,
}

struct GameState {
    players: HashMap<String, Player>,
    tagged_players: Vec<String>,
}

impl GameState {
    // Helper method to get current game status
    fn get_status(&self) -> GameStatus {
        GameStatus {
            tagged_players: self.tagged_players.clone(),
            active_players: self.players.keys().cloned().collect(),
        }
    }

    // Helper method to broadcast state to all players
    async fn broadcast_state(&self) {
        let status = self.get_status();
        let update = GameMessage::StateUpdate(status);
        
        for player in self.players.values() {
            if let Err(e) = player.ws_sender.send(update.clone()) {
                eprintln!("Failed to broadcast to player: {}", e);
            }
        }
    }
}

async fn handle_message(msg: Message, player_id: &str, state: &GameStateRef) -> Result<(), String> {
    if let Message::Text(text) = msg {
        let game_message: GameMessage = serde_json::from_str(&text)
            .map_err(|_| "Invalid message format")?;

        match game_message {
            GameMessage::Tag { tagger_id, tagged_id } => {
                let mut state = state.lock().await;
                
                // Validation checks
                if !state.players.contains_key(&tagged_id) {
                    return Err("Tagged player doesn't exist".to_string());
                }
                if tagger_id != player_id {
                    return Err("You can only send tags for yourself".to_string());
                }
                
                // Update game state
                if let Some(tagged_player) = state.players.get_mut(&tagged_id) {
                    tagged_player.is_tagged = true;
                    if !state.tagged_players.contains(&tagged_id) {
                        state.tagged_players.push(tagged_id.clone());
                    }

                    // Broadcast new state to all players
                    state.broadcast_state().await;
                }
            }
            _ => return Err("Invalid message type".to_string()),
        }
    }
    Ok(())
}

async fn handle_connection(ws: WebSocket, state: GameStateRef) {
    let (mut ws_tx, mut ws_rx) = ws.split();
    let (broadcast_tx, mut broadcast_rx) = broadcast::channel(100);

    if let Some(Ok(message)) = ws_rx.next().await {
        if let Ok(player_id) = message.to_str() {
            // Add player to game state
            {
                let mut state = state.lock().await;
                state.players.insert(
                    player_id.to_string(),
                    Player {
                        id: player_id.to_string(),
                        is_tagged: false,
                        ws_sender: broadcast_tx.clone(),
                    },
                );

                // Send initial game state to the new player
                let initial_status = state.get_status();
                let initial_message = GameMessage::StateUpdate(initial_status);
                if let Err(e) = ws_tx.send(Message::Text(
                    serde_json::to_string(&initial_message).unwrap()
                )).await {
                    eprintln!("Failed to send initial state: {}", e);
                    return;
                }

                // Broadcast to all players that a new player joined
                state.broadcast_state().await;
            }

            // Handle messages in a loop
            loop {
                tokio::select! {
                    Some(result) = ws_rx.next() => {
                        match result {
                            Ok(msg) => {
                                if let Err(e) = handle_message(msg, &player_id, &state).await {
                                    let error_msg = GameMessage::Error { 
                                        message: e 
                                    };
                                    let _ = ws_tx.send(Message::Text(
                                        serde_json::to_string(&error_msg).unwrap()
                                    )).await;
                                }
                            }
                            Err(_) => break,
                        }
                    }
                    Ok(update) = broadcast_rx.recv() => {
                        let msg = Message::Text(serde_json::to_string(&update).unwrap());
                        if let Err(_) = ws_tx.send(msg).await {
                            break;
                        }
                    }
                }
            }

            // Player disconnected - update and broadcast new state
            {
                let mut state = state.lock().await;
                state.players.remove(&player_id);
                state.broadcast_state().await;
            }
        }
    }
}