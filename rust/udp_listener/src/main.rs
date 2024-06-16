use std::net::UdpSocket;
use std::str;

fn main() -> std::io::Result<()> {
    // Bind the UDP socket to an address and port
    let socket = UdpSocket::bind("0.0.0.0:12345")?;
    println!("Listening on 0.0.0.0:12345");

    let mut buf = [0; 10000];

    loop {
        // Receive data from the socket
        let (amt, src) = socket.recv_from(&mut buf)?;

        // Convert the received bytes into a string, if possible
        let received = match str::from_utf8(&buf[..amt]) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("Invalid UTF-8 sequence: {}", e);
                continue;
            }
        };
        // Print the received message
        //println!("plopping on 0.0.0.0:12345");
        println!("Received from {}: {}", src, received);
    }
}
