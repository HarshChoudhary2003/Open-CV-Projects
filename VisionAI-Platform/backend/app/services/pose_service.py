"""
VisionAI Platform - Pose & Gesture Detection Service
Uses MediaPipe Holistic for body pose, hand gestures, and face mesh.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
    _HAS_MP = True
except ImportError:
    _HAS_MP = False


# ── Gesture vocabulary ────────────────────────────────────────────────────────
GESTURE_LABELS = {
    "thumbs_up": "👍 Thumbs Up",
    "thumbs_down": "👎 Thumbs Down",
    "open_hand": "✋ Open Hand",
    "fist": "✊ Fist",
    "peace": "✌️ Peace",
    "pointing": "👆 Pointing",
    "ok": "👌 OK",
    "unknown": "❓ Unknown",
}


@dataclass
class PoseResult:
    landmarks_2d: List[Tuple[int, int]] = field(default_factory=list)
    gesture: str = "unknown"
    pose_label: str = "standing"
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "gesture": self.gesture,
            "gesture_label": GESTURE_LABELS.get(self.gesture, "Unknown"),
            "pose_label": self.pose_label,
            "confidence": round(self.confidence, 3),
            "landmark_count": len(self.landmarks_2d),
        }


class PoseService:
    _instance: Optional["PoseService"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialised = False
        return cls._instance

    def initialise(self):
        if self._initialised:
            return

        if not _HAS_MP:
            print("[PoseService] MediaPipe not installed. Pose disabled.")
            self.enabled = False
            self._initialised = True
            return

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.enabled = True
        print("[PoseService] MediaPipe Holistic ready.")
        self._initialised = True

    # ── Main processing ───────────────────────────────────────────

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[PoseResult]]:
        if not self.enabled:
            return frame, None

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self.holistic.process(rgb)
        rgb.flags.writeable = True
        annotated = frame.copy()

        if results.pose_landmarks is None:
            return annotated, None

        # Draw skeleton
        self.mp_drawing.draw_landmarks(
            annotated,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
        )

        # Draw hand landmarks
        for hand_lm in [results.left_hand_landmarks, results.right_hand_landmarks]:
            if hand_lm:
                self.mp_drawing.draw_landmarks(
                    annotated, hand_lm, self.mp_holistic.HAND_CONNECTIONS
                )

        # Extract 2D landmarks
        lm_list = [
            (int(lm.x * w), int(lm.y * h))
            for lm in results.pose_landmarks.landmark
        ]

        pose_label = self._classify_pose(results.pose_landmarks.landmark)
        gesture = self._classify_gesture(results.right_hand_landmarks or
                                         results.left_hand_landmarks)

        pr = PoseResult(
            landmarks_2d=lm_list,
            gesture=gesture,
            pose_label=pose_label,
            confidence=0.9,
        )

        # HUD text
        cv2.putText(annotated, f"Pose: {pose_label}", (10, h - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.putText(annotated, f"Gesture: {GESTURE_LABELS.get(gesture, gesture)}",
                    (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

        return annotated, pr

    # ── Classifiers ───────────────────────────────────────────────

    def _classify_pose(self, landmarks) -> str:
        """Rule-based pose classifier using key landmark positions."""
        if not landmarks:
            return "unknown"
        # MediaPipe indices: 0=nose, 11=L_shoulder, 12=R_shoulder,
        #                    23=L_hip, 24=R_hip, 25=L_knee, 26=R_knee
        try:
            nose_y = landmarks[0].y
            hip_y = (landmarks[23].y + landmarks[24].y) / 2
            knee_y = (landmarks[25].y + landmarks[26].y) / 2
            shoulder_y = (landmarks[11].y + landmarks[12].y) / 2

            if abs(nose_y - hip_y) < 0.2:
                return "lying_down"
            if knee_y < hip_y - 0.05:
                return "crouching"
            if shoulder_y < 0.35:
                return "raising_arms"
            return "standing"
        except Exception:
            return "standing"

    def _classify_gesture(self, hand_landmarks) -> str:
        if hand_landmarks is None:
            return "none"
        lm = hand_landmarks.landmark
        try:
            # Tip indices: thumb=4, index=8, middle=12, ring=16, pinky=20
            tips = [lm[4], lm[8], lm[12], lm[16], lm[20]]
            mcp = [lm[2], lm[5], lm[9], lm[13], lm[17]]
            extended = [tips[i].y < mcp[i].y for i in range(5)]

            # All fingers extended
            if all(extended[1:]):
                return "open_hand"
            # No fingers extended
            if not any(extended[1:]):
                return "fist"
            # Only index extended
            if extended[1] and not any(extended[2:]):
                return "pointing"
            # Index + middle extended
            if extended[1] and extended[2] and not extended[3] and not extended[4]:
                return "peace"
            # Thumb + index making circle (OK)
            thumb_tip = lm[4]
            index_tip = lm[8]
            dist = ((thumb_tip.x - index_tip.x) ** 2 +
                    (thumb_tip.y - index_tip.y) ** 2) ** 0.5
            if dist < 0.05 and not any(extended[2:]):
                return "ok"
            # Thumb up
            if extended[0] and not any(extended[1:]):
                return "thumbs_up" if lm[4].y < lm[2].y else "thumbs_down"

            return "unknown"
        except Exception:
            return "unknown"

    def cleanup(self):
        if self.enabled and hasattr(self, "holistic"):
            self.holistic.close()


def get_pose_service() -> PoseService:
    svc = PoseService()
    svc.initialise()
    return svc
