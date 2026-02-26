# lerobot_teleoperator_yamactiveleader
Active YAM teleop leader device integration for LeRobot

## Getting Started

```bash
git clone https://github.com/uynitsuj/lerobot_teleoperator_yamactiveleader
cd lerobot_teleoperator_yamactiveleader
uv pip install -e .

lerobot-find-port # find out which port the WaveShare Bus Servo Adapter is on, set as arg for setup_motors.py

uv run setup_motors.py tty.usbmodem5AE60805531

lerobot-calibrate --teleop.type=yam_active_leader \
    --teleop.port=/dev/tty.usbmodem5AE60805531
    
```

## Development

Install the package in editable mode:
```bash
git clone https://github.com/SpesRobotics/lerobot-teleoperator-teleop.git
cd lerobot-teleop
uv pip install -e .
```