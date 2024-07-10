use pyo3::prelude::*;
use std::time::Instant;
use std::thread;
use std::time::Duration;
use std::net::UdpSocket;
use pyo3::wrap_pyfunction;
use std::str;
use byteorder::{ByteOrder, LittleEndian};

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyclass]
struct UdpSender {
    socket: UdpSocket,
}


#[derive(Debug, FromPyObject)]
#[pyclass]
struct ScambiUnitLedOnly {
    colour: Vec<u8>,
    physical_led_pos: Vec<u16>,
}

#[pymethods]
impl UdpSender {
    #[new]
    fn new() -> PyResult<Self> {
        let socket = UdpSocket::bind("0.0.0.0:0").map_err(PyErr::new::<pyo3::exceptions::PyOSError, _>)?;
        Ok(UdpSender { socket })
    }

    fn send_message(&self, message: &str, address: &str) -> PyResult<()> {
        self.socket.send_to(message.as_bytes(), address).map_err(PyErr::new::<pyo3::exceptions::PyOSError, _>)?;
        Ok(())
    }
    fn test_custom_obj(&self, scambiunits:Vec<ScambiUnitLedOnly>) -> PyResult<()> {
        for led_unit in &scambiunits{
            println!("Unit details: {:?}", led_unit);
        }
        let mut output_payload = Vec::new();
        let udp_delimiter = b"\xAB\xCD\xEF";
        for scambiunit in scambiunits {
            let mut pos_bytes = vec![0u8; scambiunit.physical_led_pos.len() * 2];
            LittleEndian::write_u16_into(&scambiunit.physical_led_pos, &mut pos_bytes);
    
            output_payload.extend_from_slice(&pos_bytes);
            output_payload.push(udp_delimiter); // Delimiter between pos and colour
            output_payload.extend_from_slice(&scambiunit.colour);
            output_payload.push(udp_delimiter); // Delimiter between units
        }
        Ok(())
    }

}

/// A Python module implemented in Rust.
#[pymodule]
fn led_sender(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let _socket = UdpSocket::bind("0.0.0.0:12345").expect("Failed to bind socket");
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_class::<UdpSender>()?;
    Ok(())
}
