//actviate the venv
//then use maturin develop --release in the project folder
//if built scucesfully, this module should now be available as a normal
//pyhon module in the venv context

use pyo3::prelude::*;
use std::time::Instant;
use std::thread;
use std::time::Duration;
use std::net::{UdpSocket, SocketAddr};
use pyo3::wrap_pyfunction;
use std::str;
use socket2::{Socket, Domain, Type};
use byteorder::{ByteOrder, LittleEndian};
use std::io::Write;
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
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, None)?;

        //socket.set_send_buffer_size(1)?;
        //socket.set_recv_buffer_size(1)?;
        let addr: SocketAddr = "0.0.0.0:12345".parse().unwrap();
        socket.bind(&addr.into())?;
        let socket: UdpSocket = socket.into();
        println!("Listening on: {}", socket.local_addr()?);
        Ok(UdpSender { socket })
    }

    fn send_message(&self, message: &str, address: &str) -> PyResult<()> {
        self.socket.send_to(message.as_bytes(), address).map_err(PyErr::new::<pyo3::exceptions::PyOSError, _>)?;
        Ok(())
    }

    fn send_message_bytes(&self, message: &[u8], address: &str) -> PyResult<()> {
        self.socket.send_to(message, address).map_err(PyErr::new::<pyo3::exceptions::PyOSError, _>)?;
        Ok(())
    }

    fn send_udp_scambis(&self, scambiunits: Vec<ScambiUnitLedOnly>, address: &str) -> PyResult<()> {
        let udp_delimiter = b"\xAB\xCD\xEF";
        let unit_size = scambiunits[0].physical_led_pos.len() * 2 + udp_delimiter.len() * 2 + scambiunits[0].colour.len();
        let total_size = unit_size * scambiunits.len();
        
        let mut output_payload = Vec::with_capacity(total_size);
        
        for scambiunit in scambiunits {
            let mut pos_bytes = vec![0u8; scambiunit.physical_led_pos.len() * 2];
            LittleEndian::write_u16_into(&scambiunit.physical_led_pos, &mut pos_bytes);
            
            output_payload.write_all(&pos_bytes).unwrap();
            output_payload.write_all(udp_delimiter).unwrap();
            output_payload.write_all(&scambiunit.colour).unwrap();
            output_payload.write_all(udp_delimiter).unwrap();
        }

        self.socket.send_to(&output_payload, address)
            .map_err(PyErr::new::<pyo3::exceptions::PyOSError, _>)?;
    
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