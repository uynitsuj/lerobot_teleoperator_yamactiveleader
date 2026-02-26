"""Standalone setup_motors script for YamActiveLeader teleoperator."""

import sys

import lerobot.robots  # noqa: F401 — must be imported first to resolve circular import in lerobot

from lerobot_teleoperator_yamactiveleader import (
    YamActiveLeaderTeleoperatorConfig,
    YamActiveLeaderTeleoperator,
)


def main():
    port = sys.argv[1] if len(sys.argv) > 1 else "/dev/tty.usbmodem5AE60805531"
    config = YamActiveLeaderTeleoperatorConfig(port=port)
    device = YamActiveLeaderTeleoperator(config)
    device.setup_motors()


if __name__ == "__main__":
    main()
