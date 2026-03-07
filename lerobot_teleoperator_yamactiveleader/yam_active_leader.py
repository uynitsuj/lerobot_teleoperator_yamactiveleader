import logging
import time
from typing import Any
import copy

import numpy as np
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_yam_active_leader import YamActiveLeaderTeleoperatorConfig

logger = logging.getLogger(__name__)


class YamActiveLeaderTeleoperator(Teleoperator):
    config_class = YamActiveLeaderTeleoperatorConfig
    name = "yam_active_leader_teleoperator"

    def __init__(self, config: YamActiveLeaderTeleoperatorConfig, active_trigger: bool = True):
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
        self.gripper_reading_raw = None
        self.trigger_closing = None
        self.active_trigger = active_trigger # whether to set trigger motor as an active virtual spring

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

    # --- drive_to_zero ---

    # Seconds to wait after commanding all motors to zero before releasing arm torque.
    ZERO_SETTLE_TIME: float = 2.0

    # --- drive_to_config (DAgger hold approach) ---

    # Seconds to wait after commanding the arm to the target config before
    # sampling baseline currents.  Must be long enough for the arm to fully
    # reach and settle at the target; increase for slow or loaded moves.
    DRIVE_SETTLE_TIME: float = 3.0

    # Torque limit (0–1000) applied to all arm motors during the approach
    # move.  Lower values produce a slower, gentler motion; higher values
    # snap to the target faster.  The limit persists during the subsequent
    # hold phase, so it also governs how hard the arm resists disturbances
    # while waiting for human intervention.
    DRIVE_TORQUE_LIMIT: int = 700

    # --- start_arm_hold (intervention detection) ---

    # Number of Present_Current readings averaged to establish the per-joint
    # baseline after the arm has settled.  More samples give a more stable
    # baseline at the cost of a longer setup pause (~HOLD_BASELINE_SAMPLES ×
    # HOLD_BASELINE_INTERVAL seconds).
    HOLD_BASELINE_SAMPLES: int = 30

    # Sleep time (seconds) between baseline current samples.  Together with
    # HOLD_BASELINE_SAMPLES this controls the total baseline collection time.
    HOLD_BASELINE_INTERVAL: float = 0.02

    # Raw current delta (above per-joint baseline) on any single joint that
    # is considered evidence of active human input.  Too low → false triggers
    # from noise; too high → sluggish detection.  Tune empirically by
    # watching the `max_delta` debug log while the arm is held still vs.
    # while you push against it.
    HOLD_DELTA_THRESHOLD: float = 15.0

    # EMA smoothing factor for live current readings during hold monitoring
    # (0–1).  Lower values apply heavier smoothing, reducing noise at the
    # cost of added detection lag.
    HOLD_FILTER_ALPHA: float = 0.1

    # Number of consecutive control-loop frames that must exceed
    # HOLD_DELTA_THRESHOLD before intervention is confirmed.  Acts as a
    # debounce: prevents transient current spikes from triggering a false
    # intervention.
    HOLD_SUSTAINED_FRAMES: int = 5

    # --- start_gripper_spring ---

    # Normalized target position the gripper motor drives toward when the
    # user is not squeezing (-100 … 100 in RANGE_M100_100 mode, 0 = midpoint
    # / open position after homing).
    GRIPPER_OPEN_POS: float = 0.0

    # Torque limit (0–1000) applied while the motor is driving the trigger
    # back toward the open position.  Higher values make the return snap feel
    # stronger.
    GRIPPER_RETURN_TORQUE: int = 400

    # Torque limit (0–1000) applied while the user is actively squeezing the
    # trigger.  Keep very low so the trigger is easy to hold against the
    # motor's return force.
    GRIPPER_SQUEEZE_TORQUE: int = 1

    # Displacement from GRIPPER_OPEN_POS (in normalized units) below which
    # the motor is always in return mode regardless of current.  Prevents the
    # spring logic from activating when the trigger is nearly at rest.
    GRIPPER_PRESS_THRESHOLD: float = 50.0

    # Present_Current reading (raw) above which the motor is considered to be
    # fighting the user's grip (squeeze state).  Tune with live readings from
    # read_gripper_spring_state().  Must be > GRIPPER_RELEASE_CURRENT.
    GRIPPER_SQUEEZE_CURRENT: int = 10

    # Present_Current reading (raw) below which the motor considers the user
    # to have released the trigger (release state).  Hysteresis gap between
    # this and GRIPPER_SQUEEZE_CURRENT prevents chatter at the boundary.
    GRIPPER_RELEASE_CURRENT: float = 5.0

    # EMA smoothing factor for gripper current readings (0–1).  Lower values
    # apply heavier smoothing to damp the rapid spikes that cause bang-bang
    # oscillation between squeeze and release states.
    GRIPPER_FILTER_ALPHA: float = 0.05

    def drive_to_zero(self, settle_time: float | None = None) -> None:
        """Enable torque and command every motor to the physical zero config.

        Uses raw 2048 (the encoder value that ``set_half_turn_homings``
        maps to zero config).  This is correct regardless of whether
        the calibrated range is symmetric.

        After *settle_time* seconds the arm joints are released (torque
        disabled) so the operator can move them freely.  The gripper is
        **not** released — call :pymeth:`release_gripper` explicitly if
        needed.
        """
        settle_time = settle_time if settle_time is not None else self.ZERO_SETTLE_TIME
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
        open_pos: float | None = None,
        return_torque: int | None = None,
        min_return_torque: int | None = None,
        press_threshold: float | None = None,
        p_coeff: int | None = None,
        current_squeeze_threshold: int | None = None,
        current_release_threshold: float | None = None,
        current_filter_alpha: float | None = None,
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
            open_pos: Target in normalized units (-100 … 100).
            return_torque: Torque limit while driving back to open (0-1000).
            min_return_torque: Torque limit while user is squeezing (0-1000).
                Keep low so the trigger feels easy to press.
            press_threshold: Displacement from *open_pos* below which the
                motor is always in return mode (trigger is near home).
            p_coeff: Optional PID P-gain override written once at startup
                (0-254).  Lower values give a softer spring feel.
            current_squeeze_threshold: ``Present_Current`` reading above
                which the motor is considered to be fighting the user's grip.
                Tune with live data from ``read_gripper_spring_state``.
            current_release_threshold: ``Present_Current`` reading below
                which the motor considers the user to have released the
                trigger.  Must be < *current_squeeze_threshold*.
            current_filter_alpha: EMA smoothing factor for current readings
                (0-1).  Lower values = heavier smoothing / more lag.
                Default 0.15 strongly damps the rapid spikes that cause
                bang-bang oscillation between squeeze and release states.
        """
        open_pos = open_pos if open_pos is not None else self.GRIPPER_OPEN_POS
        return_torque = return_torque if return_torque is not None else self.GRIPPER_RETURN_TORQUE
        min_return_torque = min_return_torque if min_return_torque is not None else self.GRIPPER_SQUEEZE_TORQUE
        press_threshold = press_threshold if press_threshold is not None else self.GRIPPER_PRESS_THRESHOLD
        current_squeeze_threshold = current_squeeze_threshold if current_squeeze_threshold is not None else self.GRIPPER_SQUEEZE_CURRENT
        current_release_threshold = current_release_threshold if current_release_threshold is not None else self.GRIPPER_RELEASE_CURRENT
        current_filter_alpha = current_filter_alpha if current_filter_alpha is not None else self.GRIPPER_FILTER_ALPHA
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

    def update_gripper_spring(self) -> None:
        """Update gripper torque based on current-sensing spring logic.

        Call once per control loop iteration with the current (normalized)
        gripper position from ``get_action()``.

        Reads ``Present_Current`` to detect whether the user is actively
        pressing against the motor or has released.  Modulates
        ``Torque_Limit`` without ever disabling torque, so the goal position
        always pulls the trigger toward open.
        """
        gripper_pos = self.gripper_reading_raw
        cfg = getattr(self, "_gripper_spring", None)
        if cfg is None:
            return

        displacement = abs(gripper_pos - cfg["open_pos"])
        raw_current = self.bus.sync_read("Present_Current", "gripper", normalize=False, num_retry=3)["gripper"]

        # Low-pass filter (EMA) to smooth noisy current spikes that cause
        # bang-bang oscillation between squeeze and release states.
        alpha = cfg["current_filter_alpha"]
        cfg["filtered_current"] += alpha * (raw_current - cfg["filtered_current"])
        current = cfg["filtered_current"]
        # Track displacement direction: positive delta = trigger closing (squeezing)
        disp_delta = displacement - cfg["prev_displacement"]
        cfg["prev_displacement"] = displacement
        self.trigger_closing = disp_delta > 0

        if self.trigger_closing:
            # User is actively squeezing — reduce torque
            torque = cfg["squeeze_torque"]
        else:
            torque = cfg["return_torque"]


        self.bus.sync_write("Torque_Limit", {"gripper": torque}, num_retry=3)
        logger.debug(
            "Gripper spring: pos=%.1f  disp=%.1f  disp_delta=%.2f  raw=%d  filtered=%.1f  torque=%d",
            gripper_pos, displacement, disp_delta, raw_current, current, torque,
        )

    def release_gripper(self) -> None:
        """Disable torque on the gripper so it can be moved freely."""
        self.bus.disable_torque("gripper")

    # ------------------------------------------------------------------ #
    # DAgger intervention support
    # ------------------------------------------------------------------ #

    def drive_to_config(
        self,
        joint_positions_deg: dict[str, float],
        settle_time: float | None = None,
        torque_limit: int | None = None,
    ) -> None:
        """Enable torque on arm motors and drive to the given joint positions.

        Args:
            joint_positions_deg: Dict mapping motor name → target in degrees
                (normalized degrees matching the calibrated range).
                Only arm motors are commanded; gripper is unaffected.
            settle_time: Seconds to wait for the arm to reach the target.
                Defaults to :pyattr:`DRIVE_SETTLE_TIME`.
            torque_limit: Torque limit (0–1000) applied before moving.
                Lower values produce a slower, gentler approach.  The limit
                persists into the subsequent hold phase so the arm doesn't
                snap hard against disturbances during hold either.
                Defaults to :pyattr:`DRIVE_TORQUE_LIMIT`.
        """
        settle_time = settle_time if settle_time is not None else self.DRIVE_SETTLE_TIME
        torque_limit = torque_limit if torque_limit is not None else self.DRIVE_TORQUE_LIMIT
        logger.info("Driving arm to config: %s  torque_limit=%d", {m: f"{v:.1f}" for m, v in joint_positions_deg.items()}, torque_limit)
        for motor in self.ARM_MOTORS:
            self.bus.write("Torque_Limit", motor, torque_limit)
        self.bus.enable_torque(self.ARM_MOTORS)
        for motor in self.ARM_MOTORS:
            if motor in joint_positions_deg:
                self.bus.write("Goal_Position", motor, joint_positions_deg[motor])
        time.sleep(settle_time)
        logger.info("Arm settled at target config.")

    def start_arm_hold(
        self,
        baseline_samples: int | None = None,
        baseline_interval: float | None = None,
        delta_threshold: float | None = None,
        filter_alpha: float | None = None,
        sustained_frames: int | None = None,
    ) -> None:
        """Begin monitoring for human intervention while holding current pose.

        Must be called after :pymeth:`drive_to_config` has settled.  Samples
        arm motor currents to build a per-joint baseline, then arms the
        intervention detector.

        Args:
            baseline_samples: Number of current readings to average for the
                per-joint baseline.  More samples → more stable baseline.
            baseline_interval: Sleep time between baseline samples (seconds).
            delta_threshold: Raw current delta above baseline on any joint
                that signals active human input.  Needs empirical tuning
                (start around 15 and adjust based on observed noise floor).
            filter_alpha: EMA smoothing factor for live current readings
                (0-1).  Lower = heavier smoothing / more lag.
            sustained_frames: Consecutive frames above threshold required to
                confirm intervention (debounces current noise spikes).
        """
        baseline_samples = baseline_samples if baseline_samples is not None else self.HOLD_BASELINE_SAMPLES
        baseline_interval = baseline_interval if baseline_interval is not None else self.HOLD_BASELINE_INTERVAL
        delta_threshold = delta_threshold if delta_threshold is not None else self.HOLD_DELTA_THRESHOLD
        filter_alpha = filter_alpha if filter_alpha is not None else self.HOLD_FILTER_ALPHA
        sustained_frames = sustained_frames if sustained_frames is not None else self.HOLD_SUSTAINED_FRAMES
        logger.info("Sampling baseline arm currents (%d samples)...", baseline_samples)
        accum: dict[str, float] = {m: 0.0 for m in self.ARM_MOTORS}
        for _ in range(baseline_samples):
            readings = self.bus.sync_read("Present_Current", normalize=False)
            for m in self.ARM_MOTORS:
                accum[m] += float(readings[m])
            time.sleep(baseline_interval)
        baseline = {m: accum[m] / baseline_samples for m in self.ARM_MOTORS}
        logger.info(
            "Arm hold baseline currents: %s",
            {m: f"{v:.1f}" for m, v in baseline.items()},
        )
        self._arm_hold: dict[str, Any] = {
            "baseline": baseline,
            "filtered": dict(baseline),  # start filter at baseline
            "delta_threshold": delta_threshold,
            "filter_alpha": filter_alpha,
            "sustained_frames": sustained_frames,
            "sustained_count": 0,
            "intervening": False,
        }

    @property
    def is_arm_hold_active(self) -> bool:
        """True while an arm hold is armed and intervention not yet detected."""
        return hasattr(self, "_arm_hold") and self._arm_hold is not None and not self._arm_hold["intervening"]

    @property
    def is_arm_hold_intervening(self) -> bool:
        """True once human intervention has been detected on the arm."""
        return hasattr(self, "_arm_hold") and self._arm_hold is not None and self._arm_hold["intervening"]

    def update_arm_hold(self) -> bool:
        """Check for human intervention; call once per control loop tick.

        Reads ``Present_Current`` for all arm joints, applies EMA filtering,
        computes delta from per-joint baseline, and checks for a sustained
        threshold crossing.  On detection, arm torque is disabled and hold
        state is latched.

        Returns:
            True on the first frame intervention is detected; False otherwise.
        """
        cfg = getattr(self, "_arm_hold", None)
        if cfg is None or cfg["intervening"]:
            return False

        try:
            readings = self.bus.sync_read("Present_Current", normalize=False, num_retry=3)
        except Exception as e:
            logger.warning("update_arm_hold: current read failed: %s", e)
            return False

        alpha = cfg["filter_alpha"]
        max_delta = 0.0
        for m in self.ARM_MOTORS:
            raw = float(readings[m])
            cfg["filtered"][m] += alpha * (raw - cfg["filtered"][m])
            delta = abs(cfg["filtered"][m] - cfg["baseline"][m])
            max_delta = max(max_delta, delta)

        if max_delta > cfg["delta_threshold"]:
            cfg["sustained_count"] += 1
        else:
            cfg["sustained_count"] = 0

        logger.debug(
            "Arm hold: max_delta=%.1f  sustained=%d/%d",
            max_delta, cfg["sustained_count"], cfg["sustained_frames"],
        )

        if cfg["sustained_count"] >= cfg["sustained_frames"]:
            cfg["intervening"] = True
            self.bus.disable_torque(self.ARM_MOTORS, num_retry=5)
            logger.info("Human intervention detected — arm torque released.")
            return True

        return False

    def clear_arm_hold(self) -> None:
        """Reset hold state so a new DAgger cycle can begin."""
        self._arm_hold = None

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        start = time.perf_counter()
        # instead of retrying sync read, just repeat last if failed
        try:
            action = self.bus.sync_read("Present_Position")
        except Exception as e:
            logger.error(f"Error reading action: {e}")
            action = {f"{motor}.pos": val for motor, val in self._last_action.items()}
            return action
        self._last_action = action
        action = {f"{motor}.pos": val for motor, val in action.items()}
        self.gripper_reading_raw = copy.deepcopy(action["gripper.pos"])
        if self.active_trigger:
            self.update_gripper_spring()
        action["gripper.pos"] = np.clip(1 - ((action["gripper.pos"] - 5) / (85 - 5)), 0, 1)  # normalize gripper position to 0-1 range, w/ some deadzone
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