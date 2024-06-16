use std::net::UdpSocket;
use std::str;

#[derive(Debug)]
struct ScambiUnitLedOnly {
    colour: Vec<u8>,
    physical_led_pos: Vec<u16>,
}
const UDP_DELIMITER: [u8; 3] = [0xAB, 0xCD, 0xEF];
fn main() -> std::io::Result<()> {
    // Bind the UDP socket to an address and port
    let socket = UdpSocket::bind("0.0.0.0:12345")?;
    println!("Listening on 0.0.0.0:12345");

    let mut buf = [0; 10000];
    
    loop {
        // Receive data from the socket
        let (amt, src) = socket.recv_from(&mut buf)?;
        for byte in &buf[..amt] {
            print!("{}  ", byte);
        }
        let message = &buf[..amt];

        let mut parts = Vec::new();
        let mut start = 0;

        for i in 0..message.len() {
            if i + UDP_DELIMITER.len() <= message.len() &&
               &message[i..i + UDP_DELIMITER.len()] == &UDP_DELIMITER[..]
            {
                if start < i {
                    parts.push(&message[start..i]);
                }
                start = i + UDP_DELIMITER.len();
            }
        }

        if start < message.len() {
            parts.push(&message[start..]);
        }

        // let parts: Vec<&[u8]> = message.split(|window| window == &UDP_DELIMITER[..]).collect();
        // println!();  // New line after printing bytes
        // //println!("Received {} bytes fromf {}: {:?}", amt, src, &buf[..amt]);
        // // Convert the received bytes into a string, if possible
        // let received = match str::from_utf8(&buf[..amt]) {
        //     Ok(v) => v,
        //     Err(e) => {
        //         eprintln!("Invalid UTF-8 sequence: {}", e);
        //         continue;
        //     }
        // };
        // Print the received message
        //println!("plopping on 0.0.0.0:12345");
        for item in &parts {
            println!("Part: {:?}", item);
        }
        println!("Received something whoo");
    }
}
