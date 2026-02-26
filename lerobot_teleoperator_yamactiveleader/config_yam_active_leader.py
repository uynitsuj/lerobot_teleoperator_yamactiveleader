from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("yam_active_leader")
@dataclass
class YamActiveLeaderTeleoperatorConfig(TeleoperatorConfig):
    # Port to connect to the Feetech motor bus
    port: str = "/dev/tty.usbmodem5AE60805531"

    # Whether to use degrees for angles
    use_degrees: bool = True