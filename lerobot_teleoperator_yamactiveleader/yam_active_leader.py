import logging
import time
from typing import Any

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_yam_active_leader import YamActiveLeaderTeleoperatorConfig

logger = logging.getLogger(__name__)


class YamActiveLeaderTeleoperator(Teleoperator):
    config_class = YamActiveLeaderTeleoperatorConfig
    name = "yam_active_leader_teleoperator"

    def __init__(self, config: YamActiveLeaderTeleoperatorConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "joint_1": Motor(1, "sts3215", norm_mode_body),
                "joint_2": Motor(2, "sts3215", norm_mode_body),
                "joint_3": Motor(3, "sts3215", norm_mode_body),
                "joint_4": Motor(4, "sts3215", norm_mode_body),
                "joint_5": Motor(5, "sts3215", norm_mode_body),
                "joint_6": Motor(6, "sts3215", norm_mode_body),
                "gripper": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
            },
        )

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to zero cfg and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        unknown_range_motors = [motor for motor in self.bus.motors]
        print(
            "Move all joints sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)

        # The DEGREES / RANGE normalizations map 0 to mid = (min+max)/2.
        # If the recorded range isn't symmetric around 2048, zero cfg
        # (which is at raw 2048 after homing) won't read as 0°.
        # Fix: mirror the recorded range around 2048 for every motor so
        # that mid == 2048 and zero cfg always reads 0° / 0 normalized.
        half_turn = 2048
        for motor in self.bus.motors:
            dist = max(abs(range_maxes[motor] - half_turn), abs(half_turn - range_mins[motor]))
            range_mins[motor] = half_turn - dist
            range_maxes[motor] = half_turn + dist

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def configure(self) -> None:
        self.bus.disable_torque()
        self.bus.configure_motors()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    # ------------------------------------------------------------------ #
    # Active motor control helpers
    # ------------------------------------------------------------------ #

    ARM_MOTORS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]

    HALF_TURN_RAW = 0  # raw value that always equals zero cfg after homing

    def drive_to_zero(self, settle_time: float = 2.0) -> None:
        """Enable torque and command every motor to the physical zero config.

        Uses raw 2048 (the encoder value that ``set_half_turn_homings``
        maps to zero config).  This is correct regardless of whether
        the calibrated range is symmetric.

        After *settle_time* seconds the arm joints are released (torque
        disabled) so the operator can move them freely.  The gripper is
        **not** released — call :pymeth:`release_gripper` explicitly if
        needed.
        """
        logger.info("Driving all motors to zero config (raw 2048)...")
        # --- Debug: BEFORE ---
        raw_before = self.bus.sync_read("Present_Position", normalize=False)
        # --- Enable + command ---
        self.bus.enable_torque()
        for motor in self.bus.motors:
            self.bus.write(
                "Goal_Position", motor, self.HALF_TURN_RAW, normalize=True
            )
        time.sleep(settle_time)
        # --- Debug: AFTER ---
        raw_after = self.bus.sync_read("Present_Position", normalize=False)
        norm_after = self.bus.sync_read("Present_Position")
        goal_readback = self.bus.sync_read("Goal_Position", normalize=False)
        homing = self.bus.sync_read("Homing_Offset", normalize=False)
        mins = self.bus.sync_read("Min_Position_Limit", normalize=False)
        maxs = self.bus.sync_read("Max_Position_Limit", normalize=False)
   
        for m, cal in (self.bus.calibration or {}).items():
            print(f"  {m}: range=[{cal.range_min}, {cal.range_max}] homing={cal.homing_offset}")
        # --- End debug ---
        # Release arm joints so the user can back-drive them
        self.bus.disable_torque(self.ARM_MOTORS)
        logger.info("Arm joints released — gripper still held.")

    def read_gripper_spring_state(self) -> None:
        """Read and print gripper motor state in-place (does not affect torque)."""
        torque_limit = self.bus.read("Torque_Limit", "gripper", normalize=False)
        present_position = self.bus.read("Present_Position", "gripper")
        present_velocity = self.bus.read("Present_Velocity", "gripper")
        present_load = self.bus.read("Present_Load", "gripper", normalize=False)
        present_current = self.bus.read("Present_Current", "gripper", normalize=False)
        present_temperature = self.bus.read("Present_Temperature", "gripper", normalize=False)
        print(
            f"pos:{present_position:7.2f}  vel:{present_velocity:7.2f}  "
            f"load:{present_load:5}  current:{present_current:5}  "
            f"torque_lim:{torque_limit:5}  temp:{present_temperature:3}°C",
            end="\r",
            flush=True,
        )

    def start_gripper_spring(
        self,
        open_pos: float = 0,
        return_torque: int = 400,
        min_return_torque: int = 1,
        press_threshold: float = 50,
        p_coeff: int | None = None,
        current_squeeze_threshold: int = 10,
        current_release_threshold: float = 5.0,
        current_filter_alpha: float = 0.05,
    ) -> None:
        """Set up a current-sensing adaptive spring on the gripper motor.

        ``Goal_Position`` is always set to *open_pos* and torque stays
        enabled.  Each call to :pymeth:`update_gripper_spring` reads
        ``Present_Current`` and modulates ``Torque_Limit``:

        * **Squeezing** (current > *current_squeeze_threshold* and trigger
          displaced beyond *press_threshold*): the motor is fighting the
          user's grip → drop ``Torque_Limit`` to *min_return_torque* so the
          trigger is easy to hold.

        * **Released** (current < *current_release_threshold* and trigger
          displaced): the user has let go → raise ``Torque_Limit`` to
          *return_torque* so the motor snaps the trigger back to open.

        Hysteresis between the two current thresholds prevents chatter at
        the squeeze/release boundary.

        Args:
            open_pos: Target in normalized units (−100 … 100).
            return_torque: Torque limit while driving back to open (0–1000).
            min_return_torque: Torque limit while user is squeezing (0–1000).
                Keep low so the trigger feels easy to press.
            press_threshold: Displacement from *open_pos* below which the
                motor is always in return mode (trigger is near home).
            p_coeff: Optional PID P-gain override written once at startup
                (0–254).  Lower values give a softer spring feel.
            current_squeeze_threshold: ``Present_Current`` reading above
                which the motor is considered to be fighting the user's grip.
                Tune with live data from ``read_gripper_spring_state``.
            current_release_threshold: ``Present_Current`` reading below
                which the motor considers the user to have released the
                trigger.  Must be < *current_squeeze_threshold*.
            current_filter_alpha: EMA smoothing factor for current readings
                (0–1).  Lower values = heavier smoothing / more lag.
                Default 0.15 strongly damps the rapid spikes that cause
                bang-bang oscillation between squeeze and release states.
        """
        self._gripper_spring = {
            "open_pos": open_pos,
            "return_torque": return_torque,
            "squeeze_torque": min_return_torque,
            "press_threshold": press_threshold,
            "current_squeeze_threshold": current_squeeze_threshold,
            "current_release_threshold": current_release_threshold,
            "squeezing": False,
            "current_filter_alpha": current_filter_alpha,
            "filtered_current": 0.0,
            "prev_displacement": 0.0,
        }

        # Optionally tune PID P-gain (EEPROM — needs torque off to write)
        if p_coeff is not None:
            self.bus.write("P_Coefficient", "gripper", p_coeff)

        # Set goal to open position, arm with return_torque, enable torque.
        # normalize=True maps 0 → raw 2048 (calibrated midpoint / open position).
        self.bus.write("Goal_Position", "gripper", self.HALF_TURN_RAW, normalize=True)
        self.bus.write("Torque_Limit", "gripper", return_torque)
        self.bus.enable_torque("gripper")
        logger.info(
            "Gripper spring (current-based): open_pos=%.1f  return=%d  squeeze=%d  "
            "squeeze_thresh=%d  release_thresh=%d  press_thresh=%.1f  filter_alpha=%.2f",
            open_pos, return_torque, min_return_torque,
            current_squeeze_threshold, current_release_threshold, press_threshold,
            current_filter_alpha,
        )

    def update_gripper_spring(self, gripper_pos: float) -> None:
        """Update gripper torque based on current-sensing spring logic.

        Call once per control loop iteration with the current (normalized)
        gripper position from ``get_action()``.

        Reads ``Present_Current`` to detect whether the user is actively
        pressing against the motor or has released.  Modulates
        ``Torque_Limit`` without ever disabling torque, so the goal position
        always pulls the trigger toward open.
        """
        cfg = getattr(self, "_gripper_spring", None)
        if cfg is None:
            return

        displacement = abs(gripper_pos - cfg["open_pos"])
        # raw_current = self.bus.read("Present_Current", "gripper", normalize=False)
        raw_current = self.bus.sync_read("Present_Current", "gripper", normalize=False)["gripper"]

        # Low-pass filter (EMA) to smooth noisy current spikes that cause
        # bang-bang oscillation between squeeze and release states.
        alpha = cfg["current_filter_alpha"]
        cfg["filtered_current"] += alpha * (raw_current - cfg["filtered_current"])
        current = cfg["filtered_current"]
        # Track displacement direction: positive delta = trigger closing (squeezing)
        disp_delta = displacement - cfg["prev_displacement"]
        cfg["prev_displacement"] = displacement
        closing = disp_delta > 0

        if closing:
            # User is actively squeezing — get out of the way
            torque = cfg["squeeze_torque"]
            # disable torque
            # self.bus.disable_torque("gripper")
        else:
            # Trigger is releasing or stationary — push it back open
            # self.bus.enable_torque("gripper")
            torque = cfg["return_torque"]

        # print(f"raw: {raw_current}  filtered: {current:.1f}  disp: {displacement:.1f}  "
        #       f"delta: {disp_delta:.2f}  {'closing' if closing else 'opening'}  torque: {torque}")

        self.bus.sync_write("Torque_Limit", {"gripper": torque})
        logger.debug(
            "Gripper spring: pos=%.1f  disp=%.1f  disp_delta=%.2f  raw=%d  filtered=%.1f  torque=%d",
            gripper_pos, displacement, disp_delta, raw_current, current, torque,
        )

    def release_gripper(self) -> None:
        """Disable torque on the gripper so it can be moved freely."""
        self.bus.disable_torque("gripper")

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        action = self.bus.sync_read("Present_Position")
        action = {f"{motor}.pos": val for motor, val in action.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # TODO: Implement force feedback
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        if hasattr(self, "_debug_csv_file"):
            self._debug_csv_file.close()
            print(f"[gripper_spring] debug log saved → {self._debug_csv_path}")
        self.bus.disconnect()
        logger.info(f"{self} disconnected.")