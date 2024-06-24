https://crates.io/crates/ws2818-rgb-led-spi-driver

example use here - https://github.com/phip1611/ws2818-rgb-led-spi-driver.git
install rust here (raspberry pi) : https://www.rust-lang.org/tools/install

for SPI - sudo raspi-config -> interfaces -> enable SPI

check with sudo raspi-config nonint get_spi (0=enabled??)

then plug MOSI into DIN (pin 19 on raspbnerry pi zero)

check current SPI buffer size: cat /sys/module/spidev/parameters/bufsiz

ok have to increase SPI buffer or breakup the call