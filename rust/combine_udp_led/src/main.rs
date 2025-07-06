//! Example that definitely works on Raspberry Pi.
//! Make sure you have "SPI" on your Pi enabled and that MOSI-Pin is connected (GPIO:10)
//! with DIN-Pin. You just need DIN pin, no clock. WS2818 uses one-wire-protocol.
//! See the specification for details
use std::time::Instant;
use ws2818_rgb_led_spi_driver::adapter_gen::WS28xxAdapter;
use ws2818_rgb_led_spi_driver::adapter_spi::WS28xxSpiAdapter;
use ws2818_rgb_led_spi_driver::encoding::encode_rgb;
use std::thread;
use std::time::Duration;
use std::str;
use socket2::{Socket, Domain, Type};
use std::io::Result;
use rand::Rng;
use std::net::{UdpSocket, SocketAddr};
#[derive(Debug)]
struct ScambiUnitLedOnly {
    colour: Vec<u8>,
    physical_led_pos: Vec<u16>,
}
const UDP_DELIMITER: [u8; 3] = [0xAB, 0xCD, 0xEF];



// #[inline(always)]
// pub fn sleep_busy_waiting_ms(ms: u64) {
//     // need to burn CPU cycles or it messes up timings
//     let target_time = Instant::now().add(Duration::from_millis(ms));
//     loop {
//         if Instant::now() >= target_time {
//             break;
//         }
//     }
// }

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

fn flatten_vec_to_slice(vec: &Vec<&[u8]>) -> Vec<u8> {
    vec.iter().flat_map(|slice| slice.iter().cloned()).collect()
}



fn main() -> std::io::Result<()> {
    println!("make sure you have \"SPI\" on your Pi enabled and that MOSI-Pin is connected with DIN-Pin!");
    let mut adapter = WS28xxSpiAdapter::new("/dev/spidev0.0").unwrap();
    let num_leds: u32 = 300;

    //first put a colour on the LEDS to indicate this element is operational:

    let mut green_display_bytes = vec![];
    for _ in 0..num_leds {
        green_display_bytes.extend_from_slice(&encode_rgb(0, 255, 0));
    }
    adapter.write_encoded_rgb(&green_display_bytes).unwrap();
    //bear in mind sleeps can mess up the SPI
    thread::sleep(Duration::from_secs(1));
    let mut blue_display_bytes = vec![];
    for _ in 0..num_leds {
        blue_display_bytes.extend_from_slice(&encode_rgb(0, 0, 255));
    }
    adapter.write_encoded_rgb(&blue_display_bytes).unwrap();



    // Bind the UDP socket to an address and port
    //let socket = UdpSocket::bind("0.0.0.0:12345")?;
    // Create a socket2::Socket
    // Create a socket2::Socket
    let socket = Socket::new(Domain::IPV4, Type::DGRAM, None)?;

    // Set send buffer size to 0
    socket.set_send_buffer_size(0)?;

    // Set receive buffer size to 0
    socket.set_recv_buffer_size(0)?;

    // Bind the socket2::Socket
    let addr: SocketAddr = "0.0.0.0:12345".parse().unwrap();
    socket.bind(&addr.into())?;

    // Convert socket2::Socket to std::net::UdpSocket
    let socket: UdpSocket = socket.into();

    // Print the port we're actually listening on
    println!("Listening on: {}", socket.local_addr()?);

    let mut buf = [0; 1000];
    
    //let mut led_output_vec = vec![0; 300];
    static DEFAULT_COLOR: [u8; 3] = [0, 0, 0];
    loop {
        let mut led_units: Vec<ScambiUnitLedOnly> = Vec::new();
        
        let mut led_output_vec: Vec<&[u8]> = vec![&DEFAULT_COLOR; 300];
        let (amt, src) = socket.recv_from(&mut buf)?;
        //println!("Received {} bytes from {}", amt, src);
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

        //let mut rng = rand::thread_rng();
        //this is bad - but at this point we can't run on windows PC so use this for now
        let mut display_bytes: Vec<u8> = Vec::new();
        for (i, led_colour) in led_output_vec.iter().enumerate() {
            display_bytes.extend_from_slice(&encode_rgb(
                led_colour[2],
                led_colour[1],
                led_colour[0]
            ));

        }
        // for (i, led_unit) in led_output_vec.iter().enumerate(){
        //     println!("led_unit details: {:}, {:?}",i, led_unit);
        // }
        // for (i, led_unit) in display_bytes.iter().enumerate(){
        //     println!("display_bytes details: {:}, {:?}",i, led_unit);
        // }
        // for led_unit in &led_units{
        //     println!("Unit details: {:?}", led_unit);
        // }
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
        if rand::thread_rng().gen_bool(0.0001) {
            println!("Time elapsed decoding: {:?}", duration);
        }
        let duration = start.elapsed();
        adapter.write_encoded_rgb(&display_bytes).unwrap();
        if rand::thread_rng().gen_bool(0.0001) {
            println!("Time elapsed setting leds: {:?}", duration);
        }
        
        //thread::sleep(Duration::from_millis(500));
    }
}
