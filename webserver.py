import base64
import cv2
import numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit
import mediapipe as mp
from state_manager import StateManager
from insert import both

# -----------------------------
# Flask + SocketIO
# -----------------------------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

state_manager = StateManager()


@app.route("/")
def home():
    return "OK - Flask is running"

@socketio.on("connect")
def on_connect():
    print("✅ Client connected via Socket.IO")

# -----------------------------
# Mediapipe setup
# -----------------------------
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

KEY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_ELBOW,
    mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE
]

# -----------------------------
# Utility functions
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

def calculateScore(low: int, high:int, actual: int):
    if high == low:
        return 10
    fitted_actual = max(low, min(actual, high))
    return 10 + (fitted_actual - low) * (90 / (high - low))

def reverseCalculateScore(low: int, high: int, actual: int):
    if high == low:
        return 10
    fitted_actual = max(low, min(actual, high))
    return max(100 - (fitted_actual - low) * (90 / (high - low)), 10)

@socketio.on("connect")
def handle_connect():
    sid = request.sid
    state_manager.init_state(sid)
    print(f"[CONNECT] User {sid} connected.")

@socketio.on("disconnect")
def handle_disconnect():
    sid = request.sid
    state_manager.remove_state(sid)
    print(f"[DISCONNECT] User {sid} disconnected.")

id = 2

frames_arr = []

@socketio.on("frame")
def process_frame(data):
    sid = request.sid
    state = state_manager.get_state(sid)
    if not state:
        return
    global frames_arr
    global id

    img_data = data.get("image")
    frames_arr.append(img_data)
    if(len(frames_arr)%30==0):
        both(id, "User", frames_arr)

    if not img_data:
        emit("update", {"error": "No image data"}, to=sid)
        return

    if img_data.startswith("data:image"):
        img_data = img_data.split(",")[1]

    np_arr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if frame is None:
        emit("update", {"error": "Could not decode frame"}, to=sid)
        return

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if not results.pose_landmarks or not is_pose_visible(results.pose_landmarks.landmark):
        return

    landmarks = results.pose_landmarks.landmark
    angles = extract_joint_angles(landmarks)

    # ------------------------
    # Squats
    # ------------------------
    if state["exercise"] == "Squats":
        knee = min(angles["knee_l"], angles["knee_r"])

        if state["rep_stage"] != "down" and knee <= 110:
            state["rep_stage"] = "down"
            state["min_knee_angle"] = knee
        elif state["rep_stage"] == "down" and knee <= 110:
            state["min_knee_angle"] = min(state["min_knee_angle"], knee)
        elif state["rep_stage"] == "down" and knee > 110:
            state["rep_stage"] = "up"
            state["rep_count"] += 1

            score = reverseCalculateScore(low=30, high=50, actual=state["min_knee_angle"])

            # Detailed feedback ranges
            if state["min_knee_angle"] <= 30:
                feedback = "Great squat"
            elif state["min_knee_angle"] <= 40:
                feedback = "Good squat, bend slightly more"
            elif state["min_knee_angle"] <= 50:
                feedback = "Ok squat, focus on form"
            else:
                feedback = "Bad squat, go lower"

            state["rep_scores"].append(score)

            emit("update", {
                "exercise": "Squats",
                "rep_count": state["rep_count"],
                "score": int(score),
                "feedback": feedback
            }, to=sid)


    # ------------------------
    # Push-ups
    # ------------------------
    elif state["exercise"] == "Push-ups":
        elbow_angle = min(angles["elbow_l"], angles["elbow_r"])

        if state["rep_stage"] != "down" and elbow_angle <= 90:
            state["rep_stage"] = "down"
            state["min_elbow_angle"] = elbow_angle
        elif state["rep_stage"] == "down" and elbow_angle <= 100:
            state["min_elbow_angle"] = min(state["min_elbow_angle"], elbow_angle)
        elif state["rep_stage"] == "down" and elbow_angle >= 150:
            state["rep_stage"] = "up"
            state["rep_count"] += 1

            # Detailed feedback ranges
            if state["min_elbow_angle"] <= 70:
                feedback = "Good push-up"
                score = 100
            elif state["min_elbow_angle"] <= 80:
                feedback = "Decent push-up"
                score = 80
            elif state["min_elbow_angle"] <= 100:
                feedback = "Mediocre push-up"
                score = 60
            else:
                feedback = "Poor push-up"
                score = 50

            state["rep_scores"].append(score)

            emit("update", {
                "exercise": "Push-ups",
                "rep_count": state["rep_count"],
                "score": int(score),
                "feedback": feedback
            }, to=sid)

    # ------------------------
    # Jumping Jacks
    # ------------------------
    elif state["exercise"] == "Jumping Jacks":
        hand_height = min(
            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y
        )
        knee_dist = abs(
            landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x -
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x
        )

        UP_THRESHOLD = 0.38
        DOWN_THRESHOLD = 0.45

        state["max_knee_dist"] = max(state["max_knee_dist"], knee_dist)
        state["min_knee_dist"] = min(state["min_knee_dist"], knee_dist)

        if hand_height < UP_THRESHOLD and state["rep_stage"] != "up":
            state["rep_stage"] = "up"
        elif hand_height > DOWN_THRESHOLD and state["rep_stage"] == "up":
            state["rep_stage"] = "down"
            state["rep_count"] += 1

            score = calculateScore(0.12, 0.3, state["max_knee_dist"])

            if state["max_knee_dist"] >= 0.30:
                feedback = "Good form"
            elif state["max_knee_dist"] >= 0.18:
                feedback = "Decent form"
            elif state["max_knee_dist"] >= 0.12:
                feedback = "Shallow form"
            else:
                feedback = "Very shallow form"

            state["rep_scores"].append(score)

            # Reset knee distances
            state["max_knee_dist"] = 0
            state["min_knee_dist"] = 1

            emit("update", {
                "exercise": "Jumping Jacks",
                "rep_count": state["rep_count"],
                "score": int(score),
                "feedback": feedback
            }, to=sid)


    # ------------------------
    # Lunges
    # ------------------------
    elif state["exercise"] == "Lunges":
        def knee_angle(hip, knee, ankle):
            a = np.array([hip.x, hip.y])
            b = np.array([knee.x, knee.y])
            c = np.array([ankle.x, ankle.y])
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        left_knee_angle = knee_angle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        right_knee_angle = knee_angle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        knee_used = min(left_knee_angle, right_knee_angle)

        if state["rep_stage"] != "down" and knee_used < 120:
            state["rep_stage"] = "down"
            state["min_knee_angle"] = knee_used
        elif state["rep_stage"] == "down" and knee_used < 120:
            state["min_knee_angle"] = min(state["min_knee_angle"], knee_used)
        elif state["rep_stage"] == "down" and knee_used >= 120:
            state["rep_stage"] = "up"
            state["rep_count"] += 1

            score = reverseCalculateScore(low=60, high=80, actual=state["min_knee_angle"])
            if state["min_knee_angle"] <= 60:
                feedback = "Good lunge"
            elif state["min_knee_angle"] <= 70:
                feedback = "Decent lunge"
            elif state["min_knee_angle"] <= 80:
                feedback = "Shallow lunge"
            else:
                feedback = "Very shallow lunge"

            state["rep_scores"].append(score)

            emit("update", {
                "exercise": "Lunges",
                "rep_count": state["rep_count"],
                "score": int(score),
                "feedback": feedback
            }, to=sid)

    # Update state at end of processing
    state_manager.update_state(sid, state)

@socketio.on("set_exercise")
def set_exercise(data):
    sid = request.sid
    state = state_manager.get_state(sid)
    if not state:
        return

    exercise = data.get("exercise")
    if exercise not in ["Push-ups", "Jumping Jacks", "Squats", "Lunges"]:
        emit("set_exercise_response", {"success": False, "error": "Invalid exercise"}, to=sid)
        return

    state.update({
        "exercise": exercise,
        "rep_count": 0,
        "rep_stage": None,
        "frame_scores": [],
        "rep_scores": [],
        "min_knee_angle": 180,
        "min_elbow_angle": 180
    })
    state_manager.update_state(sid, state)
    emit("set_exercise_response", {"success": True, "exercise": exercise}, to=sid)

if __name__ == "__main__":
    print("Starting RepRadar on port 5032...")
    socketio.run(app, host="0.0.0.0", port=5032, allow_unsafe_werkzeug=True)
