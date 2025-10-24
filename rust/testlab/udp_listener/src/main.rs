use std::net::UdpSocket;
use std::str;
use std::time::Instant;
#[derive(Debug)]
struct ScambiUnitLedOnly {
    colour: Vec<u8>,
    physical_led_pos: Vec<u16>,
}
const UDP_DELIMITER: [u8; 3] = [0xAB, 0xCD, 0xEF];


fn split_message_by_delimiter<'a>(message: &'a [u8], delimiter: &[u8]) -> Vec<&'a [u8]> {
    let mut parts = Vec::new();
    let mut start = 0;

    // Iterate over the message, finding all occurrences of the delimiter
    while let Some(pos) = message[start..].windows(delimiter.len()).position(|window| window == delimiter) {
        // If we find the delimiter, push the slice from the start to the position
        parts.push(&message[start..start + pos]);
        start += pos + delimiter.len();
    }

    // Add the final part after the last delimiter, if any
    if start < message.len() {
        parts.push(&message[start..]);
    }

    parts
}


fn decode_as_u16(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks(2).map(|chunk| {
        if chunk.len() == 2 {
            u16::from_le_bytes([chunk[0], chunk[1]])
        } else {
            u16::from_le_bytes([chunk[0], 0])
        }
    }).collect()
}


fn main() -> std::io::Result<()> {
    // Bind the UDP socket to an address and port
    let socket = UdpSocket::bind("0.0.0.0:12345")?;
    println!("Listening on 0.0.0.0:12345");

    let mut buf = [0; 10000];
    
    //let mut led_output_vec = vec![0; 300];
    
    loop {
        let mut led_units: Vec<ScambiUnitLedOnly> = Vec::new();
        static DEFAULT_COLOR: [u8; 3] = [0, 0, 0];
        let mut led_output_vec: Vec<&[u8]> = vec![&DEFAULT_COLOR; 300];
        let (amt, src) = socket.recv_from(&mut buf)?;
        let start = Instant::now();
        // for byte in &buf[..amt] {
        //     print!("{}  ", byte);
        // }
        let message = &buf[..amt];

        let parts = split_message_by_delimiter(&message, &UDP_DELIMITER);


        for i in 0..parts.len() / 2 {
            let physical_led_pos_start = i * 2;
            let colour_start = (i * 2) + 1;
            let u8_values: Vec<u8> = parts[colour_start].to_vec();
            //println!("Part {}: {:?}", i, u8_values);
            let unit = ScambiUnitLedOnly {
                colour:parts[colour_start].to_vec(),
                physical_led_pos:decode_as_u16(parts[physical_led_pos_start]),
            };
    
            led_units.push(unit);
        }


        // now write them into the vector
        for led_unit in &led_units {
            for &pos in &led_unit.physical_led_pos {
                if (pos as usize) < led_output_vec.len() {
                    led_output_vec[pos as usize] = &led_unit.colour;
                }
            }
        }

        for (i, led_unit) in led_output_vec.iter().enumerate(){
            println!("led_unit details: {:}, {:?}",i, led_unit);
        }
        // for led_unit in &led_units{
        //     println!("Unit details: {:?}", led_unit);
        // }
        // for (i, part) in parts.iter().enumerate() {
        //     if i % 2 == 1 {
        //         let u8_values: Vec<u8> = part.to_vec();
        //         println!("cols {}: {:?}", i, u8_values);
        //     } else {
        //         let u16_values = decode_as_u16(part);
        //         println!("positions {}: {:?}", i, u16_values);
        //     }
        // }
        let duration = start.elapsed();
        println!("Time elapsed in the code section: {:?}", duration);
        println!("Received something whoo");
    }
}
