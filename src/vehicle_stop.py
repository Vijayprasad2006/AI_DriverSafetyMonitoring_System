"""
DriveGuard AI - Vehicle Stop Request Simulation
Simulates an autonomous or assisted safe vehicle pullover sequence.
IMPORTANT: This is a research simulation prototype only.
"""

import time
import logging
from typing import Dict, Any

logger = logging.getLogger("DriveGuard.VehicleStop")

class VehicleStopSimulation:
    """Simulates multi-stage safe vehicle deceleration and stop workflow."""

    def __init__(self):
        self.is_active = False
        self.stage = "IDLE"  # IDLE, HAZARDS_ON, DECELERATING, STOPPED
        self.start_time = 0.0
        self.simulated_speed_kmh = 80.0
        self.target_speed_kmh = 0.0
        self.last_update_time = 0.0
        self.history = []

    def trigger_stop_request(self, reason: str, risk_score: float) -> Dict[str, Any]:
        """Initiates the simulated vehicle stopping protocol."""
        self.is_active = True
        self.stage = "HAZARDS_ON"
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.simulated_speed_kmh = 80.0

        event_record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": "STOP_REQUESTED",
            "reason": reason,
            "risk_score": risk_score,
            "initial_speed": self.simulated_speed_kmh
        }
        self.history.append(event_record)
        logger.warning(f"Vehicle Stop Simulation triggered: {reason}")
        return self.get_status()

    def update_simulation(self) -> Dict[str, Any]:
        """Progresses vehicle deceleration physics across elapsed time."""
        if not self.is_active:
            return self.get_status()

        now = time.time()
        elapsed = now - self.start_time
        
        # Staged simulation progression
        if elapsed < 2.0:
            self.stage = "HAZARDS_ON"
            self.simulated_speed_kmh = 80.0
        elif elapsed < 6.0:
            self.stage = "LANE_PULLOVER"
            # Decelerate linearly from 80 to 30 km/h
            progress = (elapsed - 2.0) / 4.0
            self.simulated_speed_kmh = max(30.0, 80.0 - (50.0 * progress))
        elif elapsed < 10.0:
            self.stage = "DECELERATING"
            # Decelerate from 30 to 0 km/h
            progress = (elapsed - 6.0) / 4.0
            self.simulated_speed_kmh = max(0.0, 30.0 - (30.0 * progress))
        else:
            self.stage = "STOPPED_SAFE"
            self.simulated_speed_kmh = 0.0

        self.last_update_time = now
        return self.get_status()

    def cancel_stop_request(self) -> Dict[str, Any]:
        """Driver manual override / cancellation of stop protocol."""
        self.is_active = False
        self.stage = "OVERRIDDEN"
        self.simulated_speed_kmh = 80.0
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Returns the current state of vehicle stop simulation."""
        return {
            "is_active": self.is_active,
            "stage": self.stage,
            "speed_kmh": round(self.simulated_speed_kmh, 1),
            "hazard_lights": self.is_active,
            "pulled_over": self.stage == "STOPPED_SAFE",
            "elapsed_seconds": round(time.time() - self.start_time, 1) if self.is_active else 0.0
        }
