import base64
import cv2
import numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit
import mediapipe as mp
import threading
import queue
from insert import both

# -----------------------------
# Flask + WebSocket Setup
# -----------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


@app.route("/")
def home():
    return "OK - Flask is running"

@socketio.on("connect")
def on_connect():
    print("✅ Client connected via Socket.IO")

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
min_knee_angle = None
min_elbow_angle = 999
max_knee_dist = 0

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
LUNGE_ANGLE_THRESHOLD = 120

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

# ------------------------
# Linear Scoring
# ------------------------
def linear_score(value, good_threshold, bad_threshold, invert=False):
    if invert:
        if value <= good_threshold:
            return 100
        elif value >= bad_threshold:
            return 10
        else:
            return 10 + (bad_threshold - value) / (bad_threshold - good_threshold) * 90
    else:
        if value >= good_threshold:
            return 100
        elif value <= bad_threshold:
            return 10
        else:
            return 10 + (value - bad_threshold) / (good_threshold - bad_threshold) * 90

# -----------------------------
# WebSocket Frame Handler
# -----------------------------
last_pose_visible = True
id = 2

frames_arr = []

@socketio.on("frame")
def process_frame(data):
    global rep_count, rep_stage, frame_scores, rep_scores, current_exercise
    global last_pose_visible, min_knee_angle, min_elbow_angle, max_knee_dist
    global frames_arr
    global id

    img_data = data.get("image")
    frames_arr.append(img_data)
    if(len(frames_arr)%30==0):
        both(id, "User", frames_arr)

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

    frame_score = 100
    angles = {}

    pose_visible = results.pose_landmarks and is_pose_visible(results.pose_landmarks.landmark)

    if pose_visible:
        landmarks = results.pose_landmarks.landmark
        angles = extract_joint_angles(landmarks)

        # ------------------------
        # Exercise Logic
        # ------------------------
        if current_exercise == "Squats":
            knee = min(angles["knee_l"], angles["knee_r"])

            # Reset min_knee_angle at start of rep
            if rep_stage != "down":
                min_knee_angle = 180

            # Down phase
            if knee <= 110 and rep_stage != "down":
                rep_stage = "down"
                min_knee_angle = knee
            elif knee <= 110 and rep_stage == "down":
                min_knee_angle = min(min_knee_angle, knee)

            # Up phase & rep finishes
            elif knee > 110 and rep_stage == "down":
                rep_stage = "up"
                rep_count += 1
                frame_scores.clear()

                # Linear scoring (smaller angle = better)
                score = linear_score(min_knee_angle, good_threshold=70, bad_threshold=110, invert=True)
                feedback = "Good squat" if score > 50 else "Shallow squat"
                rep_scores.append(score)

                emit("update", {
                    "rep_count": rep_count,
                    "score": int(score),
                    "feedback": feedback,
                    "exercise": current_exercise
                })

                print(f"[Squat] Rep {rep_count}: {feedback} | min knee {min_knee_angle:.1f}")

            frame_scores.append(100)

        elif current_exercise == "Push-ups":
            global min_elbow_angle

            elbow_l = angles["elbow_l"]
            elbow_r = angles["elbow_r"]
            elbow_angle = min(elbow_l, elbow_r)

            # Reset at start of new rep
            if rep_stage != "down":
                min_elbow_angle = 999

            # Down stage
            if elbow_angle <= 90 and rep_stage != "down":
                rep_stage = "down"
                min_elbow_angle = elbow_angle
            elif elbow_angle <= 90 and rep_stage == "down":
                min_elbow_angle = min(min_elbow_angle, elbow_angle)

            # Up stage and rep finishes
            elif elbow_angle >= 150 and rep_stage == "down":
                rep_stage = "up"
                rep_count += 1

                # Scoring
                score = linear_score(min_elbow_angle, good_threshold=70, bad_threshold=150, invert=True)
                feedback = "Good push-up" if score > 50 else "Bad push-up"
                rep_scores.append(score)

                emit("update", {
                    "rep_count": rep_count,
                    "score": int(score),
                    "feedback": feedback,
                    "exercise": current_exercise
                })

                print(f"[Push-up] Rep {rep_count}: {feedback} | min angle {min_elbow_angle:.1f}")

            # UI overlay
            cv2.putText(frame, f"Elbow Angle: {elbow_angle:.1f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            frame_scores.append(100)


        elif current_exercise == "Jumping Jacks":
            hand_height = min(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
                              landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
            knee_dist = abs(landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x -
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)

            global max_knee_dist

            # Track max knee distance while in up stage
            if rep_stage == "up":
                max_knee_dist = max(max_knee_dist, knee_dist)

            # Detect up stage
            if knee_dist > JJ_KNEE_UP and hand_height < JJ_HAND_HEIGHT and rep_stage != "up":
                rep_stage = "up"
                max_knee_dist = knee_dist  # reset at start of rep

            # Detect down phase & rep finished
            if (knee_dist < JJ_KNEE_DOWN or hand_height > JJ_HAND_HEIGHT) and rep_stage == "up":
                rep_stage = "down"
                rep_count += 1

                MIN_DIST = 0.06  # corresponds to bad
                MAX_DIST = 0.12  # corresponds to good

                # Clamp and normalize
                dist_clamped = max(MIN_DIST, min(MAX_DIST, max_knee_dist))
                score = 10 + (dist_clamped - MIN_DIST) / (MAX_DIST - MIN_DIST) * (100 - 10)
                score = int(score)

                # Feedback based on score thresholds
                if score >= 85:
                    feedback = "Excellent form"
                elif score >= 60:
                    feedback = "Good form"
                else:
                    feedback = "Shallow jump"

                rep_scores.append(score)

                # Emit update
                emit("update", {
                    "rep_count": rep_count,
                    "score": score,
                    "feedback": feedback,
                    "exercise": current_exercise
                })

                print(f"[Jumping Jack] Rep {rep_count}: {feedback} | max knee dist {max_knee_dist:.2f}")

                max_knee_dist = 0  # reset for next rep

            frame_scores.append(100)

        elif current_exercise == "Lunges":
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

            if rep_stage != "down":
                min_knee_angle = 180
            if knee_angle_used < LUNGE_ANGLE_THRESHOLD and rep_stage != "down":
                rep_stage = "down"
                min_knee_angle = knee_angle_used
            elif knee_angle_used < LUNGE_ANGLE_THRESHOLD and rep_stage == "down":
                min_knee_angle = min(min_knee_angle, knee_angle_used)
            elif knee_angle_used >= LUNGE_ANGLE_THRESHOLD and rep_stage == "down":
                rep_stage = "up"
                rep_count += 1
                frame_scores.clear()
                MIN_ANGLE = 70
                MAX_ANGLE = 80
                angle_clamped = max(MIN_ANGLE, min(MAX_ANGLE, min_knee_angle))
                score = 50 + (MAX_ANGLE - angle_clamped) / (MAX_ANGLE - MIN_ANGLE) * (100 - 50)

                feedback = "Good lunge" if score > 50 else "Shallow lunge"
                rep_scores.append(int(score))
                emit("update", {"rep_count": rep_count, "score": int(score),
                                "feedback": feedback, "exercise": current_exercise})

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
