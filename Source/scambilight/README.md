set up - run the setup file depending on scambilight unit (see folders)

this should modify the autorun file, which then should
fire off a bash script to clone repo and build rust, and ptoentially
another command to run a bootstrap file or launch the rust
care has to be taken to avoid sudo as this causes lots of problems with the cloned
repo.. note the files to help with problems




https://crates.io/crates/ws2818-rgb-led-spi-driver

example use here - https://github.com/phip1611/ws2818-rgb-led-spi-driver.git
install rust here (raspberry pi) : https://www.rust-lang.org/tools/install

for SPI - sudo raspi-config -> interfaces -> enable SPI

check with sudo raspi-config nonint get_spi (0=enabled??)

then plug MOSI into DIN (pin 19 on raspbnerry pi zero)

check current SPI buffer size: cat /sys/module/spidev/parameters/bufsiz

ok have to increase SPI buffer or breakup the call

add spidev.bufsiz=xxxx to /boot/cmdline.txt


add the following to allow SPI to run faster::
# 10% overclock
arm_freq=1100
over_voltage=8
sdram_freq=500
sdram_over_voltage=2
force_turbo=1
boot_delay=1