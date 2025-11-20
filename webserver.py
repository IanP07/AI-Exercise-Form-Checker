import base64
import cv2
import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit
import mediapipe as mp
import threading
import queue

# -----------------------------
# Flask + WebSocket Setup
# -----------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# -----------------------------
# Mediapipe Pose Setup
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# Tracker State
# -----------------------------
current_exercise = "Jumping Jacks"
rep_count = 0
rep_stage = None
frame_scores = []
rep_scores = []
lock = threading.Lock()
frame_queue = queue.Queue()
min_knee_angle = None  # Track lowest knee angle per squat/lunge rep
min_elbow_angle = 999

# -----------------------------
# Thresholds
# -----------------------------
# Push-ups
PUSHUP_DOWN, PUSHUP_UP = 90, 160

# Jumping Jacks
JJ_HAND_HEIGHT = 0.08
JJ_KNEE_UP = 0.06
JJ_KNEE_DOWN = 0.03

# Squats
SQUAT_GOOD = 70
SQUAT_MEDIOCRE = 110
SQUAT_UP = 170

# Lunges
LUNGE_ANGLE_THRESHOLD = 120  # knee angle to detect downward phase


# Key landmarks for visibility check
KEY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
]

# -----------------------------
# Helper Functions
# -----------------------------

def calculateScore(low: int, high:int, actual: int):
    """
    Linearly scale `actual` between `low` and `high` to a score from 10 to 100.
    """
    # Prevent division by zero
    if high == low:
        return 10

    # Fit actual value between low and high ranges
    fitted_actual = max(low, min(actual, high))

    # Linear scaling
    score = 10 + (fitted_actual - low) * (90 / (high - low))
    return score


def reverseCalculateScore(low: int, high: int, actual: int):
    """
    Linearly scale `actual` between `low` and `high` to a score from 10 to 100,
    where lower values are better (higher score) and higher values are worse (lower score).
    """
    # Prevent division by zero
    if high == low:
        return 10

    # Fit actual value between low and high ranges
    fitted_actual = max(low, min(actual, high))

    # Reverse linear scaling
    score = 100 - (fitted_actual - low) * (90 / (high - low))

    # Ensure minimum score is 10
    return max(score, 10)


def angle_between_points(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def extract_joint_angles(landmarks):
    def side_angles(side):
        shoulder = angle_between_points(
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y]
        )
        elbow = angle_between_points(
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ELBOW").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_WRIST").value].y]
        )
        hip = angle_between_points(
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_SHOULDER").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].y]
        )
        knee = angle_between_points(
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_HIP").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_KNEE").value].y],
            [landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].x,
             landmarks[getattr(mp_pose.PoseLandmark, f"{side}_ANKLE").value].y]
        )
        return shoulder, elbow, hip, knee

    shoulder_l, elbow_l, hip_l, knee_l = side_angles("LEFT")
    shoulder_r, elbow_r, hip_r, knee_r = side_angles("RIGHT")

    return {
        "shoulder_l": shoulder_l, "elbow_l": elbow_l, "hip_l": hip_l, "knee_l": knee_l,
        "shoulder_r": shoulder_r, "elbow_r": elbow_r, "hip_r": hip_r, "knee_r": knee_r
    }

def is_pose_visible(landmarks, visibility_threshold=0.5, required_ratio=0.6):
    visible_count = sum([1 for lm in KEY_LANDMARKS if landmarks[lm.value].visibility >= visibility_threshold])
    return (visible_count / len(KEY_LANDMARKS)) >= required_ratio

# -----------------------------
# WebSocket Frame Handler
# -----------------------------
last_pose_visible = True

@socketio.on("frame")
def process_frame(data):
    global rep_count, rep_stage, frame_scores, rep_scores, current_exercise
    global last_pose_visible, min_knee_angle, min_elbow_angle, max_knee_dist,min_knee_dist

    img_data = data.get("image")
    if not img_data:
        emit("update", {"error": "No image data"})
        return

    if img_data.startswith("data:image"):
        img_data = img_data.split(",")[1]

    np_arr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        emit("update", {"error": "Could not decode frame"})
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    pose_visible = results.pose_landmarks and is_pose_visible(results.pose_landmarks.landmark)

    if pose_visible:
        score = 0
        landmarks = results.pose_landmarks.landmark
        angles = extract_joint_angles(landmarks)

        # ------------------------
        # Exercise Logic
        # ------------------------
        if current_exercise == "Squats":

            # Picks smaller knee between left and right
            knee = min(angles["knee_l"], angles["knee_r"])

            # If out of down phase, reset the minimum knee angle
            if rep_stage != "down":
                min_knee_angle = 180

            # If knee angle is less/equal to 110 and currently not in down phase, mark as down phase
            if knee <= 110 and rep_stage != "down":
                rep_stage = "down"
                min_knee_angle = knee

            # If the knee angle is less/equal to 110
            elif knee <= 110 and rep_stage == "down":
                min_knee_angle = min(min_knee_angle, knee)

            # If knee angle greater than 110 and in down phase, change phase to up
            elif knee > 110 and rep_stage == "down":
                rep_stage = "up"
                rep_count += 1

                # Scoring section
                score = calculateScore(low=30, high=50, actual=min_knee_angle)

                # ----- Feedback statements -----
                if min_knee_angle <= 30:
                    feedback = "Great squat"
                elif min_knee_angle <= 40:
                    feedback = "Good squat, bend slightly more"
                elif min_knee_angle <= 50:
                    feedback = "Ok squat, focus on form"
                else:  # barely bending
                    feedback = "Bad squat, go lower"

                rep_scores.append(score)

                emit("update", {
                    "rep_count": rep_count,
                    "score": int(score),
                    "feedback": feedback,
                    "exercise": current_exercise
                })

                print(f"[Squat] Rep {rep_count}: {feedback} | min knee angle {min_knee_angle:.1f}")

            frame_scores.append(score)

        elif current_exercise == "Push-ups":
            elbow_l = angles["elbow_l"]
            elbow_r = angles["elbow_r"]
            # Pick smaller elbow between left and right
            elbow_angle = min(elbow_l, elbow_r)

            # If out of down phase, reset the minimum
            if rep_stage != "down":
                min_elbow_angle = 180

            # If elbow angle is less/equal to 90 and currently not in down phase, mark as down
            if elbow_angle <= 90 and rep_stage != "down":
                rep_stage = "down"
                min_elbow_angle = elbow_angle

            # If the knee angle is less/equal to 100
            elif elbow_angle <= 100 and rep_stage == "down":
                min_elbow_angle = min(min_elbow_angle, elbow_angle)

            # If knee angle is greater than 150 and in down phase change to up
            elif elbow_angle >= 150 and rep_stage == "down":
                rep_stage = "up"
                rep_count += 1

                # Scoring section
                score = calculateScore(low=70, high=100, actual=min_knee_angle)

                if min_elbow_angle <= 70:
                    feedback = "Good push-up"
                elif min_elbow_angle <= 80:
                    feedback = "Decent push-up"
                elif min_elbow_angle <= 100:
                    feedback = "Mediocre push-up"
                else:
                    feedback = "Poor push-up"

                emit("update", {
                    "rep_count": rep_count,
                    "score": score,
                    "feedback": feedback,
                    "exercise": current_exercise
                })
                rep_scores.append(score)

                print(f"[Push-up] Rep {rep_count}: {feedback} | min angle {min_elbow_angle}")

            # UI overlay
            cv2.putText(frame, f"Elbow Angle: {elbow_angle:.1f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            frame_scores.append(score)

        elif current_exercise == "Jumping Jacks":

            hand_height = min(
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
            )
            knee_dist = abs(
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x -
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
            )

            UP_THRESHOLD = 0.38  # hands low
            DOWN_THRESHOLD = 0.45  # hands high


            # Get max and min knee distances
            max_knee_dist = max(max_knee_dist, knee_dist)
            min_knee_dist = min(min_knee_dist, knee_dist)

            # If hand height is less than up threshold and not up stage, change to up stage
            if hand_height < UP_THRESHOLD and rep_stage != "up":
                rep_stage = "up"
                max_knee_dist = knee_dist
                print("[JJ] UP detected")

            # If hand height is greater than down threshold and rep stage is up, change to down increaes rep
            elif hand_height > DOWN_THRESHOLD and rep_stage == "up":
                rep_stage = "down"
                rep_count += 1

                score = calculateScore(low=0.12, high=0.3, actual=min_knee_angle)

                if max_knee_dist >= 0.30:
                    feedback = "Good form"
                elif max_knee_dist >= 0.18:
                    feedback = "Decent form"
                elif max_knee_dist >= 0.12:
                    feedback = "Shallow form"
                else:
                    feedback = "Very shallow form"

                emit("update", {
                    "exercise": "Jumping Jacks",
                    "rep_count": rep_count,
                    "score": score,
                    "feedback": feedback
                })

                # Reset knee data for next rep
                max_knee_dist = 0
                min_knee_dist = 1

            frame_scores.append(score)


        elif current_exercise == "Lunges":

            # Calculate angles
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
            left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
            right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
            right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
            right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

            def knee_angle(hip, knee, ankle):
                a = np.array([hip.x, hip.y])
                b = np.array([knee.x, knee.y])
                c = np.array([ankle.x, ankle.y])
                ba = a - b
                bc = c - b
                cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
                return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

            left_knee_angle = knee_angle(left_hip, left_knee, left_ankle)
            right_knee_angle = knee_angle(right_hip, right_knee, right_ankle)
            knee_angle_used = min(left_knee_angle, right_knee_angle)

            # Rep stage tracking
            if rep_stage != "down":
                min_knee_angle = 180
            if knee_angle_used < LUNGE_ANGLE_THRESHOLD and rep_stage != "down":
                rep_stage = "down"
                min_knee_angle = knee_angle_used
            elif knee_angle_used < LUNGE_ANGLE_THRESHOLD and rep_stage == "down":
                min_knee_angle = min(min_knee_angle, knee_angle_used)
            # Rep completed when knee rises
            elif knee_angle_used >= LUNGE_ANGLE_THRESHOLD and rep_stage == "down":
                rep_stage = "up"
                rep_count += 1
                frame_scores.clear()

                score = reverseCalculateScore(low=60, high=80, actual=min_knee_angle)

                # Calculate lunge score based on min knee angle
                if min_knee_angle <= 60:
                    rep_score = 100
                    feedback = "Good lunge"
                elif min_knee_angle <= 70:
                    rep_score = 70
                    feedback = "Decent lunge"
                elif min_knee_angle <= 80:
                    rep_score = 60
                    feedback = "Shallow lunge"
                else:
                    rep_score = 50
                    feedback = "Very shallow lunge"

                # Store score
                rep_scores.append(rep_score)

                # Emit update
                emit("update", {
                    "rep_count": rep_count,
                    "score": int(rep_score),
                    "feedback": feedback,
                    "exercise": current_exercise
                })

                print(f"[Lunge] Rep {rep_count}: {feedback}")

            frame_scores.append(score)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # ------------------------
    # UI Overlay
    # ------------------------
    avg_score = np.mean(frame_scores) if frame_scores else 0
    cv2.putText(frame, f"Exercise: {current_exercise}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Reps: {rep_count}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Score: {int(avg_score)}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

    # Overlay knee angle for lunges
    if current_exercise == "Lunges" and pose_visible:
        cv2.putText(frame, f"Knee Angle: {knee_angle_used:.1f}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    frame_queue.put(frame)

# -----------------------------
# Change Exercise
# -----------------------------
@socketio.on("set_exercise")
def set_exercise(data):
    global current_exercise, rep_count, rep_stage, frame_scores, rep_scores
    exercise = data.get("exercise")
    if exercise not in ["Push-ups", "Jumping Jacks", "Squats", "Lunges"]:
        emit("set_exercise_response", {"success": False, "error": "Invalid exercise"})
        return
    with lock:
        current_exercise = exercise
        rep_count = 0
        rep_stage = None
        frame_scores.clear()
        rep_scores.clear()
    emit("set_exercise_response", {"success": True, "exercise": exercise})

# -----------------------------
# Run Server
# -----------------------------
def run_socketio():
    socketio.run(app, host="0.0.0.0", port=5032, allow_unsafe_werkzeug=True)

if __name__ == "__main__":
    print("Starting WebSocket Exercise Tracker on port 5032...")
    threading.Thread(target=run_socketio, daemon=True).start()

    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break
            cv2.imshow("Exercise Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()