// Type definitions that are referenced in game_types.rs
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};
use rand::Rng;


mod game_types;
use game_types::{GameMessage, GameMessagePayload, GameStatus};
use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::protocol::Message};
use futures::{StreamExt, SinkExt};
use std::net::SocketAddr;
use log::{info, error};
use tokio::sync::broadcast;
use std::sync::Arc;
use axum::{routing::get, Router};
use anyhow::Result;

// Shared state pattern for concurrent access:
// - Arc = Multiple owners across threads
// - RwLock = Multiple readers OR one writer
// - GameStatus = Our actual game data
type SharedState = Arc<tokio::sync::RwLock<GameStatus>>;

// Helper function to create new shared state
fn create_shared_game_status() -> SharedState {
    Arc::new(                           // Layer 3: Multiple ownership
        tokio::sync::RwLock::new(       // Layer 2: Thread-safe access
            GameStatus {                // Layer 1: Your actual data
                players: vec![],
            }
        )
    )
}

// Why each layer is needed:
// 
// Layer 1 - GameStatus: Your actual game data
// 
// Layer 2 - RwLock: Protects against data races
//   - Multiple tasks can read at the same time
//   - Only one task can write at a time
//   - Prevents crashes from concurrent access
//
// Layer 3 - Arc: Allows multiple ownership
//   - Each task gets its own "handle" to the same data
//   - When all tasks finish, the data is automatically cleaned up
//   - "Arc" = "Atomically Reference Counted"


async fn start_http_server() -> Result<()> {
    let app = Router::new().route("/", get(|| async { "plz go away" }));

    println!("Running on http://localhost:9001");
    axum::Server::bind(&"127.0.0.1:9001".parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

async fn start_periodic_broadcast(broadcast_tx: Arc<broadcast::Sender<Message>>, state: SharedState) {
    let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(50));
    let mut counter = 0u64;

    loop {
        interval.tick().await;
        
        // Create a placeholder GameStatus message with random health values
        let mut rng = rand::thread_rng();
        let game_status = game_types::GameStatus {
            players: vec![
                game_types::Player {
                    id: "player1".to_string(),
                    name: "Alice".to_string(),
                    health: rng.gen_range(1..=100),
                },
                game_types::Player {
                    id: "player2".to_string(),
                    name: "Bob".to_string(),
                    health: rng.gen_range(1..=100),
                },
            ],
        };

        let game_message = GameMessage {
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            payload: Some(GameMessagePayload::GameStatus(game_status)),
        };

        // Only broadcast if there are active connections
        if broadcast_tx.receiver_count() > 0 {
            // Serialize the message to JSON
            match serde_json::to_string(&game_message) {
                Ok(json_message) => {
                    counter += 1;
                    info!("Broadcasting periodic message #{counter}: GameStatus update to {} clients", broadcast_tx.receiver_count());
                    
                    // Send the message to all connected clients
                    if let Err(e) = broadcast_tx.send(Message::Text(json_message)) {
                        error!("Failed to broadcast periodic message: {e}");
                    }
                }
                Err(e) => {
                    error!("Failed to serialize periodic message: {e}");
                }
            }
        } else {
            // Optionally log that we're skipping broadcast (you might want to remove this in production)
            // info!("Skipping broadcast - no active connections");
        }
    }
}

#[tokio::main]
async fn main() {
    // Initialize the logger with info level if RUST_LOG is not set
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }
    env_logger::init();

    // Create the shared state
    let shared_state = create_shared_game_status();

    // Spawn the HTTP server task
    let http_server_handle = tokio::spawn(async {
        if let Err(e) = start_http_server().await {
            error!("HTTP server failed: {e}");
            // Optionally, return the error if needed elsewhere, though JoinHandle captures panic/completion
            // Err(e)
        }
        // Ok(()) // Or Ok if returning Result
    });

    info!("HTTP server task spawned.");

    // Spawn a separate task to monitor the HTTP server's completion status
    tokio::spawn(async move {
        match http_server_handle.await {
            Ok(_) => info!("HTTP server task completed successfully."), // This implies the server stopped gracefully or the task finished.
            Err(e) => error!("HTTP server task failed: {e}"), // This catches panics or cancellations.
        }
    });

    let addr_str = "127.0.0.1:8080";

    let addr: SocketAddr = match addr_str.parse() {
        Ok(parsed_addr) => parsed_addr,
        Err(e) => {
            error!("Failed to parse hardcoded address \"{addr_str}\" : {e}");
            std::process::exit(1); 
        }
    };

    // Create the TCP listener for WebSockets
    let listener = TcpListener::bind(&addr).await.expect("Failed to bind WebSocket listener");

    info!("WebSocket server listening on: {addr}");

    // Create a broadcast channel with a capacity of 100 messages
    let (broadcast_tx, _) = broadcast::channel(100);
    let broadcast_tx = Arc::new(broadcast_tx);

    // Spawn a task to periodically broadcast game status updates
    let periodic_broadcast_handle = tokio::spawn(start_periodic_broadcast(
        broadcast_tx.clone(), 
        shared_state.clone()
    ));
    info!("Periodic broadcast task started - sending GameStatus hf");

    // Monitor the periodic broadcast task
    tokio::spawn(async move {
        match periodic_broadcast_handle.await {
            Ok(_) => info!("Periodic broadcast task completed (this shouldn't happen)"),
            Err(e) => error!("Periodic broadcast task failed: {e}"),
        }
    });

    while let Ok((stream, _)) = listener.accept().await {
        let tx = broadcast_tx.clone();
        let state = shared_state.clone();
        tokio::spawn(handle_connection(stream, tx, state));
    }
}

async fn handle_connection(stream: TcpStream, broadcast_tx: Arc<broadcast::Sender<Message>>, state: SharedState) {
    // Accept the WebSocket connection
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            error!("Error during the websocket handshake: {e}");
            return;
        }
    };

    info!("New WebSocket connection established");

    // Split the WebSocket stream into a sender and receiver
    let (mut ws_sender, mut ws_receiver) = ws_stream.split();

    // Subscribe to the broadcast channel
    let mut broadcast_rx = broadcast_tx.subscribe();

    // Send a welcome message to the newly connected client
    if let Err(e) = ws_sender.send(Message::Text("Welcome to the WebSocket server!".to_string())).await {
        error!("Error sending welcome message: {e}");
        return;
    }

    // Task to forward broadcasted messages to the client
    //move here is to move ownership of the parameters to the closure for some reason
    let send_task = tokio::spawn(async move {
        while let Ok(message) = broadcast_rx.recv().await {
            if let Err(e) = ws_sender.send(message).await {
                error!("Error sending broadcast message to client: {e}");
                break;
            }
        }
    });

    // Task to receive messages from the client and broadcast them
    let receive_task = {
        let broadcast_tx = broadcast_tx.clone();
        tokio::spawn(async move {
            while let Some(msg) = ws_receiver.next().await {
                match msg {
                    Ok(Message::Binary(data)) => {
                        info!("Received binary message of {} bytes", data.len());

                        // Try to decode the JSON message
                        match serde_json::from_slice::<GameMessage>(&data) {
                            Ok(game_message) => {
                                // Handle the decoded message
                                if let Err(e) = handle_websocket_message(game_message).await {
                                    error!("Error handling WebSocket message: {e}");
                                }
                                
                                // Optionally broadcast the original message to other clients
                                if let Err(e) = broadcast_tx.send(Message::Binary(data)) {
                                    error!("Error broadcasting message: {e}");
                                }
                            }
                            Err(e) => {
                                error!("Failed to decode JSON message: {e}");
                                // Still broadcast the original message if decoding fails
                                if let Err(e) = broadcast_tx.send(Message::Binary(data)) {
                                    error!("Error broadcasting message: {e}");
                                }
                            }
                        }
                    }
                    Ok(Message::Text(text)) => {
                        info!("Received text message: {text}");
                        
                        // Try to parse text as JSON GameMessage
                        match serde_json::from_str::<GameMessage>(&text) {
                            Ok(game_message) => {
                                if let Err(e) = handle_websocket_message(game_message).await {
                                    error!("Error handling WebSocket message: {e}");
                                }
                            }
                            Err(e) => {
                                error!("Failed to decode JSON text message: {e}");
                            }
                        }
                        
                        // Broadcast the text message
                        if let Err(e) = broadcast_tx.send(Message::Text(text)) {
                            error!("Error broadcasting text message: {e}");
                        }
                    }
                    Ok(Message::Close(_)) => {
                        info!("Client disconnected");
                        break;
                    }
                    Ok(_) => (), // Handle other message types if needed
                    Err(e) => {
                        error!("Error receiving message: {e}");
                        break;
                    }
                }
            }
        })
    };

    // Await both tasks; if either fails, the connection is closed
    tokio::select! {
        _ = send_task => (),
        _ = receive_task => (),
    }

    info!("Connection closed");
}

async fn handle_websocket_message(message: GameMessage) -> Result<(), Box<dyn std::error::Error>> {
    println!("🎮 Received GameMessage at timestamp: {}", message.timestamp);
    
    match message.payload {
        Some(GameMessagePayload::Connection(connection)) => {
            println!("👤 Player connected: {}", 
                connection.player_name,);
            info!("Player connected: {}", 
                connection.player_name,);
        }
        Some(GameMessagePayload::Tag(tag)) => {
            println!("🏷️ Tag event: {} (health: {})", tag.id, tag.health);
            info!("Tag event: {} (health: {})", tag.id, tag.health);
        }
        Some(GameMessagePayload::TagImage(tag_image)) => {
            println!("📸 Tag image from {}: {} bytes", 
                tag_image.id, tag_image.image_data.len());
            info!("Tag image from {}: {} bytes", 
                tag_image.id, tag_image.image_data.len());
        }
        Some(GameMessagePayload::GameStatus(game_status)) => {
            println!("🎯 Game status: {} players", 
                game_status.players.len());
            info!("Game status: {} players", 
                game_status.players.len());
        }
        None => {
            println!("⚠️ Received GameMessage with no payload");
            info!("Received GameMessage with no payload");
        }
    }
    Ok(())
}