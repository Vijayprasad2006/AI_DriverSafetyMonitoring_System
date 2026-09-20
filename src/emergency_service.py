"""
DriveGuard AI - Emergency Notification Service
Handles Twilio SMS dispatch with cooldowns, event deduplication,
and graceful simulated fallback mode when credentials are not configured.
"""

import time
import os
import logging
from typing import Dict, Any, Optional
import config

logger = logging.getLogger("DriveGuard.EmergencyService")

class EmergencyService:
    """Manages emergency notifications with rate limiting, cooldowns, and simulation modes."""

    def __init__(self,
                 account_sid: Optional[str] = None,
                 auth_token: Optional[str] = None,
                 from_number: Optional[str] = None,
                 contact_phone: Optional[str] = None,
                 contact_name: Optional[str] = None,
                 enabled: Optional[bool] = None):
        
        self.account_sid = account_sid or config.TWILIO_ACCOUNT_SID
        self.auth_token = auth_token or config.TWILIO_AUTH_TOKEN
        self.from_number = from_number or config.TWILIO_FROM_NUMBER
        self.contact_phone = contact_phone or config.EMERGENCY_CONTACT_PHONE
        self.contact_name = contact_name or config.EMERGENCY_CONTACT_NAME
        self.enabled = enabled if enabled is not None else config.ENABLE_SMS_ALERTS
        
        self.cooldown_seconds = config.SMS_COOLDOWN_SECONDS
        self.last_sent_time = 0.0
        self.sent_count = 0
        self.last_status: Dict[str, Any] = {
            "status": "IDLE",
            "message": "Ready",
            "timestamp": None,
            "mode": "REAL" if self.is_configured() else "SIMULATION"
        }

    def is_configured(self) -> bool:
        """Returns True if valid Twilio credentials are provided."""
        return bool(self.account_sid and self.auth_token and self.from_number and self.contact_phone)

    def can_send(self) -> bool:
        """Checks if cooldown period has elapsed."""
        if not self.enabled:
            return False
        return (time.time() - self.last_sent_time) >= self.cooldown_seconds

    def send_emergency_alert(self, reason: str, risk_score: float, force: bool = False) -> Dict[str, Any]:
        """
        Sends an SMS alert to the registered emergency contact.
        Falls back to safe simulation mode if Twilio is unconfigured.
        """
        now = time.time()
        elapsed = now - self.last_sent_time
        
        if not force and not self.enabled:
            self.last_status = {
                "status": "DISABLED",
                "message": "SMS alerts are currently disabled in settings.",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "DISABLED"
            }
            return self.last_status

        if not force and elapsed < self.cooldown_seconds:
            remaining = int(self.cooldown_seconds - elapsed)
            self.last_status = {
                "status": "COOLDOWN",
                "message": f"Alert suppressed by cooldown ({remaining}s remaining).",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "COOLDOWN"
            }
            return self.last_status

        # Message body
        timestamp_str = time.strftime("%H:%M:%S")
        sms_body = (
            f"⚠️ DRIVEGUARD AI EMERGENCY ALERT ⚠️\n"
            f"Time: {timestamp_str}\n"
            f"Contact: {self.contact_name}\n"
            f"Driver Risk: CRITICAL ({risk_score:.0f}/100)\n"
            f"Reason: {reason}\n"
            f"Action: Vehicle stop recommended immediately."
        )

        if self.is_configured():
            # Real Twilio SMS dispatch
            try:
                from twilio.rest import Client
                client = Client(self.account_sid, self.auth_token)
                message = client.messages.create(
                    body=sms_body,
                    from_=self.from_number,
                    to=self.contact_phone
                )
                self.last_sent_time = now
                self.sent_count += 1
                self.last_status = {
                    "status": "SENT_REAL",
                    "message": f"Real SMS sent successfully (Twilio SID: {message.sid}).",
                    "sid": message.sid,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "REAL"
                }
                logger.info(f"Twilio SMS sent: {message.sid}")
            except Exception as e:
                self.last_status = {
                    "status": "FAILED",
                    "message": f"Twilio SMS failed: {str(e)}",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "mode": "REAL"
                }
                logger.error(f"Failed to dispatch SMS: {e}")
        else:
            # High-fidelity Simulation Mode
            self.last_sent_time = now
            self.sent_count += 1
            self.last_status = {
                "status": "SENT_SIMULATED",
                "message": f"[SIMULATION] SMS queued to {self.contact_phone or '+1-555-0199'} ({self.contact_name}): {reason}",
                "simulated_body": sms_body,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "SIMULATION"
            }
            logger.info(f"Simulated emergency SMS sent for: {reason}")

        return self.last_status
