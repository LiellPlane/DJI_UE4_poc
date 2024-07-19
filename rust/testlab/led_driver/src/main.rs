//! Example that definitely works on Raspberry Pi.
//! Make sure you have "SPI" on your Pi enabled and that MOSI-Pin is connected
//! with DIN-Pin. You just need DIN pin, no clock. WS2818 uses one-wire-protocol.
//! See the specification for details
//! 
//! this is to prove we can flash 300 leds fast without it corrupting
use std::time::Instant;
use ws2818_examples::{get_led_num_from_args, sleep_busy_waiting_ms};
use ws2818_rgb_led_spi_driver::adapter_gen::WS28xxAdapter;
use ws2818_rgb_led_spi_driver::adapter_spi::WS28xxSpiAdapter;
use ws2818_rgb_led_spi_driver::encoding::encode_rgb;
use rand::Rng;
use std::iter::Cycle;
use std::slice::Iter;
const FREQUENCY: u64 = 25; // in Hz

const FLASH_TIME_MS: u64 =3;

fn main() {
    let colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]];
    let mut color_cycle = colors.iter().cycle();
    println!("make sure you have \"SPI\" on your Pi enabled and that MOSI-Pin is connected with DIN-Pin!");
    let mut adapter = WS28xxSpiAdapter::new("/dev/spidev0.0").unwrap();
    let num_leds: u32 = 300;



    let mut empty_display_bytes = vec![];
    for _ in 0..num_leds {
        empty_display_bytes.extend_from_slice(&encode_rgb(0, 0, 0));
    }
    let mut rng = rand::thread_rng();
    loop {

        let mut white_display_bytes = vec![];
        let random_number = rng.gen_range(1..=255);  // This line is correct
        let color = color_cycle.next().unwrap();
        for i in 0..num_leds {
            //let led_value = std::cmp::min(i, 255);
            //let finalval = std::cmp::max(led_value, 0) as u8;
            white_display_bytes.extend_from_slice(&encode_rgb(color[0], color[1], color[2]));
        }

        adapter.write_encoded_rgb(&white_display_bytes).unwrap(); // 7ms for 300 leds
        sleep_busy_waiting_ms(FLASH_TIME_MS);
        adapter.write_encoded_rgb(&empty_display_bytes).unwrap();
        sleep_busy_waiting_ms((1000 / FREQUENCY) - FLASH_TIME_MS);
    }
}
