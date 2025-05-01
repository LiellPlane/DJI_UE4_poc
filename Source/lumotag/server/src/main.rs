use tokio::net::{TcpListener, TcpStream};
use tokio_tungstenite::{accept_async, tungstenite::protocol::Message};
use futures::{StreamExt, SinkExt};
use std::env;
use std::net::SocketAddr;
use log::{info, error};
use tokio::sync::broadcast;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use axum::{routing::get, Router};
use anyhow::Result;

struct GameState {
    // positions, scores, etc.
}
// Interior mutability with concurrency:
type SharedState = Arc<tokio::sync::RwLock<GameState>>;


async fn start_http_server() -> Result<()> {
    let app = Router::new().route("/", get(|| async { "plz go away" }));

    println!("Running on http://localhost:9001");
    axum::Server::bind(&"127.0.0.1:9001".parse()?)
        .serve(app.into_make_service())
        .await?;

    Ok(())
}

#[tokio::main]
async fn main() {
    // Initialize the logger
    env_logger::init();

    // Spawn the HTTP server task
    let http_server_handle = tokio::spawn(async {
        if let Err(e) = start_http_server().await {
            error!("HTTP server failed: {}", e);
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
            Err(e) => error!("HTTP server task failed: {}", e), // This catches panics or cancellations.
        }
    });

    let addr_str = "127.0.0.1:8080";

    let addr: SocketAddr = match addr_str.parse() {
        Ok(parsed_addr) => parsed_addr,
        Err(e) => {
            error!("Failed to parse hardcoded address \"{}\" : {}", addr_str, e);
            std::process::exit(1); 
        }
    };

    // Create the TCP listener for WebSockets
    let listener = TcpListener::bind(&addr).await.expect("Failed to bind WebSocket listener");

    info!("WebSocket server listening on: {}", addr);

    // Create a broadcast channel with a capacity of 100 messages
    let (broadcast_tx, _) = broadcast::channel(100);
    let broadcast_tx = Arc::new(broadcast_tx);

    while let Ok((stream, _)) = listener.accept().await {
        let tx = broadcast_tx.clone();
        tokio::spawn(handle_connection(stream, tx));
    }
}

async fn handle_connection(stream: TcpStream, broadcast_tx: Arc<broadcast::Sender<Message>>) {
    // Accept the WebSocket connection
    let ws_stream = match accept_async(stream).await {
        Ok(ws) => ws,
        Err(e) => {
            error!("Error during the websocket handshake: {}", e);
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
        error!("Error sending welcome message: {}", e);
        return;
    }

    // Task to forward broadcasted messages to the client
    //move here is to move ownership of the parameters to the closure for some reason
    let send_task = tokio::spawn(async move {
        while let Ok(message) = broadcast_rx.recv().await {
            if let Err(e) = ws_sender.send(message).await {
                error!("Error sending broadcast message to client: {}", e);
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
                    Ok(Message::Text(text)) => {
                        info!("Received message: {}", text);

                        // Optionally process the message (e.g., reverse it)
                        // let processed_text = text.chars().rev().collect::<String>();

                        // Broadcast the original or processed message to all clients
                        if let Err(e) = broadcast_tx.send(Message::Text(text)) {
                            error!("Error broadcasting message: {}", e);
                        }
                    }
                    Ok(Message::Close(_)) => {
                        info!("Client disconnected");
                        break;
                    }
                    Ok(_) => (), // Handle other message types if needed
                    Err(e) => {
                        error!("Error receiving message: {}", e);
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