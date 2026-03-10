# lerobot_teleoperator_yamactiveleader
Active YAM teleop leader device integration for LeRobot

<img src="media/yam_active_leader.gif" width="500">
<img src="media/yam_active_leader_dagger.gif" width="500">

[Bill of Materials](BOM.md)

[Onshape CAD](https://cad.onshape.com/documents/1bf1095238bbd48b2eeeb7b5/w/c5977f299960b20a4f0d6e0a/e/8cb256dc79a0563d72aa541c?renderMode=0&uiState=69af20032fb51000194a64a9)

[Printable STL Files](stl_files)

## Getting Started

```bash
git clone https://github.com/uynitsuj/lerobot_teleoperator_yamactiveleader
cd lerobot_teleoperator_yamactiveleader
uv pip install -e .

lerobot-find-port # find out which port the WaveShare Bus Servo Adapter is on, set as arg for setup_motors.py

uv run setup_motors.py /dev/tty.usbmodem5AE60805531

lerobot-calibrate --teleop.type=yam_active_leader --teleop.port=/dev/tty.usbmodem5AE60806691 --teleop.id=left
lerobot-calibrate --teleop.type=yam_active_leader --teleop.port=/dev/tty.usbmodem5AE60805531 --teleop.id=right
    
```

## Development

Install the package in editable mode:
```bash
git clone https://github.com/SpesRobotics/lerobot-teleoperator-teleop.git
cd lerobot-teleop
uv pip install -e .
```