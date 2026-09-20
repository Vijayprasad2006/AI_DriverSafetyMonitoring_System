"""
DriveGuard AI - Driver Risk Assessment Engine
Combines eye closure (EAR), PERCLOS, yawning (MAR), head pose angles,
and driver presence into a holistic, weighted safety score (0 - 100).
Incorporates hysteresis and temporal smoothing to avoid alert flickering.
"""

import time
from typing import Dict, Any, List, Optional
import config

class RiskEngine:
    """
    Evaluates multi-signal driver fatigue and distraction.
    Uses hysteresis thresholds and persistence buffers for stable alerting.
    """

    def __init__(self,
                 ear_threshold: float = config.EAR_DROWSY_THRESHOLD,
                 mar_threshold: float = config.MAR_YAWN_THRESHOLD,
                 perclos_warning: float = config.PERCLOS_WARNING_THRESHOLD,
                 perclos_critical: float = config.PERCLOS_CRITICAL_THRESHOLD,
                 warning_min_score: float = config.RISK_SCORE_WARNING_MIN,
                 critical_min_score: float = config.RISK_SCORE_CRITICAL_MIN):
        
        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self.perclos_warning = perclos_warning
        self.perclos_critical = perclos_critical
        self.warning_min_score = warning_min_score
        self.critical_min_score = critical_min_score

        # State tracking
        self.current_risk_level = config.RISK_LEVEL_NORMAL
        self.smoothed_risk_score = 0.0
        self.last_state_change_time = time.time()
        self.escalation_counter = 0
        self.recovery_timer = 0.0

    def compute_risk(self,
                     ear: float,
                     closed_frames: int,
                     perclos: float,
                     is_yawning: bool,
                     recent_yawns: int,
                     pitch: float,
                     yaw: float,
                     is_nodding: bool,
                     face_detected: bool,
                     missing_face_sec: float = 0.0) -> Dict[str, Any]:
        """
        Processes instantaneous indicators into a synthesized risk assessment.
        Returns:
            Dict containing raw_score, smoothed_score, risk_level, reasons, recommendation.
        """
        reasons: List[str] = []
        score = 0.0

        # 1. Face Presence Evaluation
        if not face_detected:
            if missing_face_sec > config.MAX_MISSING_FACE_SECONDS:
                face_score = config.WEIGHT_FACE_PRESENCE * min(1.0, missing_face_sec / 5.0)
                score += face_score
                reasons.append(f"Driver face not visible for {missing_face_sec:.1f}s")
        else:
            # 2. Eye Closure Duration (Active Microsleep Risk)
            if closed_frames >= config.MICROSLEEP_FRAMES_THRESHOLD:
                score += config.WEIGHT_EAR_CLOSURE + 40 # Direct elevation to critical
                reasons.append(f"Critical eye closure / Microsleep ({closed_frames} frames)")
            elif closed_frames >= config.DROWSINESS_FRAMES_CRITICAL:
                score += config.WEIGHT_EAR_CLOSURE * 0.85
                reasons.append(f"Prolonged eye closure ({closed_frames} frames)")
            elif closed_frames >= config.DROWSINESS_FRAMES_WARNING:
                score += config.WEIGHT_EAR_CLOSURE * 0.50
                reasons.append("Brief eye closure / drowsiness detected")
            elif ear < self.ear_threshold:
                score += config.WEIGHT_EAR_CLOSURE * 0.25

            # 3. Moving PERCLOS (Cumulative Fatigue)
            if perclos >= self.perclos_critical:
                score += config.WEIGHT_PERCLOS
                reasons.append(f"High cumulative eye closure (PERCLOS: {perclos*100:.1f}%)")
            elif perclos >= self.perclos_warning:
                score += config.WEIGHT_PERCLOS * 0.60
                reasons.append(f"Elevated eye closure (PERCLOS: {perclos*100:.1f}%)")

            # 4. Yawning Frequency
            if is_yawning:
                score += config.WEIGHT_YAWN_FREQUENCY * 0.70
                reasons.append("Active yawn detected")
            
            if recent_yawns >= config.YAWN_CRITICAL_RATE:
                score += config.WEIGHT_YAWN_FREQUENCY
                reasons.append(f"Frequent yawning ({recent_yawns} yawns recently)")
            elif recent_yawns >= config.YAWN_WARNING_RATE:
                score += config.WEIGHT_YAWN_FREQUENCY * 0.50
                reasons.append(f"Repeated yawning ({recent_yawns} yawns recently)")

            # 5. Head Pose & Nodding (Distraction & Head Slumping)
            head_penalty = 0.0
            if pitch < config.HEAD_PITCH_DOWN_THRESHOLD:
                head_penalty += config.WEIGHT_HEAD_POSE * 0.70
                reasons.append(f"Head slumped downward ({pitch:.1f}° pitch)")
            elif abs(yaw) > abs(config.HEAD_YAW_LEFT_THRESHOLD):
                head_penalty += config.WEIGHT_HEAD_POSE * 0.50
                reasons.append(f"Distracted gaze ({yaw:.1f}° yaw)")

            if is_nodding:
                head_penalty += config.WEIGHT_HEAD_POSE * 0.80
                reasons.append("Rhythmic head nodding / microsleep sign detected")

            score += min(config.WEIGHT_HEAD_POSE, head_penalty)

        # Cap raw score between 0 and 100
        raw_score = max(0.0, min(100.0, score))

        # Exponential Moving Average for smoothing
        # Immediate escalation for critical emergency conditions to protect driver
        if raw_score >= self.critical_min_score:
            self.smoothed_risk_score = max(self.smoothed_risk_score, raw_score)
        else:
            self.smoothed_risk_score = (
                config.EMA_ALPHA * raw_score + (1.0 - config.EMA_ALPHA) * self.smoothed_risk_score
            )

        # Target risk level based on score
        if self.smoothed_risk_score >= self.critical_min_score:
            candidate_level = config.RISK_LEVEL_CRITICAL
        elif self.smoothed_risk_score >= self.warning_min_score:
            candidate_level = config.RISK_LEVEL_WARNING
        else:
            candidate_level = config.RISK_LEVEL_NORMAL

        # Hysteresis & Persistence logic
        now = time.time()
        final_level = self.current_risk_level

        # Escalation: Fast response if candidate is more severe
        if self._level_rank(candidate_level) > self._level_rank(self.current_risk_level):
            self.escalation_counter += 1
            if self.escalation_counter >= config.RISK_ESCALATION_FRAMES or candidate_level == config.RISK_LEVEL_CRITICAL:
                final_level = candidate_level
                self.current_risk_level = final_level
                self.last_state_change_time = now
                self.escalation_counter = 0
                self.recovery_timer = 0.0
        # De-escalation: Requires sustained recovery period
        elif self._level_rank(candidate_level) < self._level_rank(self.current_risk_level):
            self.escalation_counter = 0
            if self.recovery_timer == 0.0:
                self.recovery_timer = now
            elif (now - self.recovery_timer) >= config.RISK_RECOVERY_SECONDS:
                final_level = candidate_level
                self.current_risk_level = final_level
                self.last_state_change_time = now
                self.recovery_timer = 0.0
        else:
            self.escalation_counter = 0
            self.recovery_timer = 0.0

        exact_state_info = self.predict_exact_driver_state(
            ear=ear, closed_frames=closed_frames, is_yawning=is_yawning,
            pitch=pitch, yaw=yaw, is_nodding=is_nodding,
            face_detected=face_detected, missing_sec=missing_face_sec
        )

        recommendation = self._generate_recommendation(final_level, reasons)

        return {
            "raw_score": round(raw_score, 1),
            "smoothed_score": round(self.smoothed_risk_score, 1),
            "risk_level": final_level,
            "reasons": reasons if reasons else ["Normal driver alertness maintained."],
            "recommendation": recommendation,
            "candidate_level": candidate_level,
            "exact_driver_state": exact_state_info["label"],
            "state_details": exact_state_info
        }

    def predict_exact_driver_state(self,
                                   ear: float,
                                   closed_frames: int,
                                   is_yawning: bool,
                                   pitch: float,
                                   yaw: float,
                                   is_nodding: bool,
                                   face_detected: bool,
                                   missing_sec: float = 0.0) -> Dict[str, Any]:
        """
        Determines the exact, unambiguous driver state according to ISO/TR 21959-1 standards.
        Returns exact_state, severity, iso_code, and action_required.
        """
        if not face_detected or missing_sec > 2.0:
            return {
                "state": "DRIVER_UNMONITORED",
                "label": "DRIVER UNMONITORED / OCCLUDED",
                "severity": "WARNING",
                "iso_code": "ISO-DMS-ERR-01",
                "description": f"Facial region not identified in optical field of view for {missing_sec:.1f}s."
            }
        
        if closed_frames >= config.MICROSLEEP_FRAMES_THRESHOLD:
            return {
                "state": "MICROSLEEP_EPISODE",
                "label": "MICROSLEEP EPISODE (STAGE 2)",
                "severity": "CRITICAL",
                "iso_code": "ISO-DMS-FAT-03",
                "description": f"Critical eye closure sustained for {closed_frames} consecutive frames."
            }
        
        if closed_frames >= config.DROWSINESS_FRAMES_WARNING:
            return {
                "state": "DROWSINESS_STAGE_1",
                "label": "DROWSINESS DETECTED (STAGE 1)",
                "severity": "WARNING",
                "iso_code": "ISO-DMS-FAT-02",
                "description": f"Prolonged eye closure/droop detected (EAR: {ear:.2f})."
            }
            
        if is_yawning:
            return {
                "state": "YAWNING_OCCURRENCE",
                "label": "YAWNING / RESPIRATORY FATIGUE",
                "severity": "WARNING",
                "iso_code": "ISO-DMS-FAT-01",
                "description": "Prolonged stomatal/mouth aperture exceeding 0.60 threshold."
            }
            
        if pitch < config.HEAD_PITCH_DOWN_THRESHOLD or is_nodding:
            return {
                "state": "HEAD_SLUMP_NODDING",
                "label": "HEAD SLUMP / SAGITTAL DEVIATION",
                "severity": "WARNING",
                "iso_code": "ISO-DMS-POS-02",
                "description": f"Sagittal pitch downward slump ({pitch:.1f}°) or rhythmic nodding detected."
            }
            
        if abs(yaw) > abs(config.HEAD_YAW_LEFT_THRESHOLD):
            return {
                "state": "GAZE_DISTRACTION",
                "label": "GAZE DISTRACTION / OFF-ROAD YAW",
                "severity": "WARNING",
                "iso_code": "ISO-DMS-DIS-01",
                "description": f"Transverse head yaw turned away from forward trajectory ({yaw:+.1f}°)."
            }
            
        return {
            "state": "ATTENTIVE_NOMINAL",
            "label": "ATTENTIVE (NOMINAL)",
            "severity": "NORMAL",
            "iso_code": "ISO-DMS-OK-00",
            "description": "Nominal visual alertness and forward roadway fixation maintained."
        }

    def _level_rank(self, level: str) -> int:
        ranks = {config.RISK_LEVEL_NORMAL: 0, config.RISK_LEVEL_WARNING: 1, config.RISK_LEVEL_CRITICAL: 2}
        return ranks.get(level, 0)

    def _generate_recommendation(self, level: str, reasons: List[str]) -> str:
        """Context-sensitive driver recommendations."""
        if level == config.RISK_LEVEL_CRITICAL:
            return "CRITICAL FATIGUE DETECTED: Immediately pull over at a safe location, engage parking brake, and rest."
        elif level == config.RISK_LEVEL_WARNING:
            return "WARNING: Elevated drowsiness or distraction observed. Adjust cabin ventilation and plan a rest break."
        else:
            return "NORMAL: Driver appears attentive. Continue driving safely."
