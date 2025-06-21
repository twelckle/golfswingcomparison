import cv2
import pickle
import os
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import cosine
import ffmpeg
import base64
from io import BytesIO
from PIL import Image

mp_pose = mp.solutions.pose
NAME_MAP = {
    "Aberg": "Ludvig Åberg",
    "Dechambeau": "Bryson DeChambeau",
    "DJ": "Dustin Johnson",
    "Fowler": "Rickie Fowler",
    "Matsuyama": "Hideki Matsuyama",
    "Morikawa": "Collin Morikawa",
    "Rory": "Rory McIlroy",
    "Schauffele": "Xander Schauffele",
    "Scheffler": "Scottie Scheffler",
    "Tiger": "Tiger Woods"
}

# --- Helpers ---

def get_rotation(video_path):
    try:
        meta = ffmpeg.probe(video_path)
        rotate_tag = meta['streams'][0]['tags'].get('rotate', 0)
        return int(rotate_tag)
    except Exception:
        return 0

def get_pose_vector(frame, pose_model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        coords = np.array([[lm.x, lm.y] for lm in landmarks])
        center = (coords[11] + coords[12]) / 2
        coords -= center
        coords /= np.linalg.norm(coords) + 1e-6
        return coords.flatten()
    return None

def extract_pose_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    rotation = get_rotation(video_path)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if rotation == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif rotation == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frames.append(frame)
    cap.release()
    return frames

def resample_vectors(vectors, target_length=60):
    if len(vectors) == target_length:
        return vectors
    idx = np.linspace(0, len(vectors) - 1, target_length).astype(int)
    return [vectors[i] for i in idx]

def find_closest_pro(user_frames, pose_db):
    best_match = None
    best_score = -1
    all_scores = []

    with mp_pose.Pose(static_image_mode=True) as pose_model:
        user_vectors = [get_pose_vector(f, pose_model) for f in user_frames]
        user_vectors = [v for v in user_vectors if v is not None]
        user_vectors = resample_vectors(user_vectors, 60)

        for pro in pose_db:
            pro_vectors = resample_vectors(pro["pose_vectors"], 60)
            scores = []
            for u, p in zip(user_vectors, pro_vectors):
                sim = 1 - cosine(u, p)
                scores.append(sim)
            avg_score = np.mean(scores)
            score_percent = round(avg_score * 100, 2)
            all_scores.append((pro["name"], score_percent))

            if avg_score > best_score:
                best_score = avg_score
                best_match = pro

    # 🔥 Print similarity leaderboard
    print("\n🏌️ Pro Similarity Scores:")
    for name, score in sorted(all_scores, key=lambda x: -x[1]):
        display_name = NAME_MAP.get(name, name)
        print(f"  {display_name.ljust(20)} → {score:.2f}%")

    full_name = NAME_MAP.get(best_match["name"], best_match["name"])
    return {**best_match, "name": full_name}, round(best_score * 100, 2)

def overlay_pose(frame, pose_model, center_landmark_index=0):
    black = np.zeros_like(frame)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        height, width, _ = frame.shape

        # Compute shift to center based on selected landmark
        if 0 <= center_landmark_index < len(landmarks):
            cx = int(landmarks[center_landmark_index].x * width)
            cy = int(landmarks[center_landmark_index].y * height)

            # Compute offset to move selected point to center
            offset_x = width // 2 - cx
            offset_y = height // 2 - cy

            # Translate landmarks before drawing
            translated_landmarks = []
            for lm in landmarks:
                x = int(lm.x * width + offset_x)
                y = int(lm.y * height + offset_y)
                translated_landmarks.append((x, y))

            # Draw translated skeleton
            for connection in mp_pose.POSE_CONNECTIONS:
                start_idx, end_idx = connection
                start = translated_landmarks[start_idx]
                end = translated_landmarks[end_idx]
                cv2.line(black, start, end, (255, 255, 255), 6)
                cv2.circle(black, start, 8, (0, 0, 255), -1)
                cv2.circle(black, end, 8, (0, 0, 255), -1)

    blended = cv2.addWeighted(frame, 0.2, black, 0.8, 0)
    return blended

def generate_overlay_frames(user_frames, pro_frames):
    max_height = max(f.shape[0] for f in user_frames + pro_frames)
    max_width = max(f.shape[1] for f in user_frames + pro_frames)
    canvas_size = max(max_height, max_width) + 200

    overlay_images_base64 = []

    with mp_pose.Pose(static_image_mode=True) as pose_model:
        for i in range(len(user_frames)):
            user_frame = user_frames[i]
            pro_frame = pro_frames[i]
            blank = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

            results_user = pose_model.process(cv2.cvtColor(user_frame, cv2.COLOR_BGR2RGB))
            if results_user.pose_landmarks:
                user_landmarks = results_user.pose_landmarks.landmark
                height, width, _ = user_frame.shape
                user_hips = [(user_landmarks[23].x + user_landmarks[24].x) / 2,
                             (user_landmarks[23].y + user_landmarks[24].y) / 2]
                cx = int(user_hips[0] * width)
                cy = int(user_hips[1] * height)
                offset_x = canvas_size // 2 - cx
                offset_y = canvas_size // 2 - cy

                translated = []
                for lm in user_landmarks:
                    x = np.clip(int(lm.x * width + offset_x), 0, canvas_size - 1)
                    y = np.clip(int(lm.y * height + offset_y), 0, canvas_size - 1)
                    translated.append((x, y))

                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = translated[start_idx]
                    end = translated[end_idx]
                    cv2.line(blank, start, end, (255, 255, 255), 6)
                    cv2.circle(blank, start, 8, (0, 0, 255), -1)
                    cv2.circle(blank, end, 8, (0, 0, 255), -1)

            results_pro = pose_model.process(cv2.cvtColor(pro_frame, cv2.COLOR_BGR2RGB))
            if results_pro.pose_landmarks:
                pro_landmarks = results_pro.pose_landmarks.landmark
                translated = []
                for lm in pro_landmarks:
                    x = np.clip(int(lm.x * width + offset_x), 0, canvas_size - 1)
                    y = np.clip(int(lm.y * height + offset_y), 0, canvas_size - 1)
                    translated.append((x, y))

                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start = translated[start_idx]
                    end = translated[end_idx]
                    cv2.line(blank, start, end, (255, 255, 255), 6)
                    cv2.circle(blank, start, 8, (0, 255, 0), -1)
                    cv2.circle(blank, end, 8, (0, 255, 0), -1)

            # Encode frame to base64
            pil_img = Image.fromarray(cv2.cvtColor(blank, cv2.COLOR_BGR2RGB))
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            overlay_images_base64.append(f"data:image/png;base64,{base64_img}")

    return overlay_images_base64

def frame_viewer(user_frames, pro_frames, similarity_score, pro_name):
    i = 0
    total = len(user_frames)
    with mp_pose.Pose(static_image_mode=True) as pose_model:
        user_frames = [overlay_pose(f, pose_model, center_landmark_index=23) for f in user_frames]
        pro_frames = [overlay_pose(f, pose_model, center_landmark_index=23) for f in pro_frames]

    while True:
        user_frame = user_frames[i]
        pro_frame = pro_frames[i]

        if user_frame.shape != pro_frame.shape:
            pro_frame = cv2.resize(pro_frame, (user_frame.shape[1], user_frame.shape[0]))

        combined = np.hstack((user_frame, pro_frame))

        cv2.putText(combined, "YOU", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, pro_name.upper(), (combined.shape[1] // 2 + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, f"Frame {i + 1}/{total}", (combined.shape[1] // 2 - 100, combined.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(combined, f"Similarity: {similarity_score:.2f}%", (combined.shape[1] // 2 - 150, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 215, 0), 2)

        cv2.imshow("Swing Comparison Viewer", combined)
        key = cv2.waitKey(0)

        if key == 27:
            break
        elif key == 2:
            i = max(0, i - 1)
        elif key == 3:
            i = min(total - 1, i + 1)

    cv2.destroyAllWindows()


def run_analysis(user_video_path):
    with open("pro_pose_db.pkl", "rb") as f:
        pose_db = pickle.load(f)

    user_frames = extract_pose_frames(user_video_path)
    user_frames = resample_vectors(user_frames, 60)
    best_pro, similarity = find_closest_pro(user_frames, pose_db)

    # Overlay stick figures
    with mp_pose.Pose(static_image_mode=True) as pose_model:
        overlaid_frames = []
        for frame in user_frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose_model.process(frame_rgb)
            overlay = np.zeros_like(frame)
            overlay = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width, _ = overlay.shape
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    x1 = int(landmarks[start_idx].x * width)
                    y1 = int(landmarks[start_idx].y * height)
                    x2 = int(landmarks[end_idx].x * width)
                    y2 = int(landmarks[end_idx].y * height)
                    cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.circle(overlay, (x1, y1), 6, (0, 0, 255), -1)
                    cv2.circle(overlay, (x2, y2), 6, (0, 0, 255), -1)

            overlaid_frames.append(overlay)

    # Encode user frames as base64
    frame_paths = []
    for frame in overlaid_frames:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        frame_paths.append(f"data:image/png;base64,{base64_img}")

    # Also process pro swing frames
    pro_frames = extract_pose_frames(best_pro["video_path"])
    pro_frames = resample_vectors(pro_frames, 60)
    with mp_pose.Pose(static_image_mode=True) as pose_model:
        pro_overlaid = []
        for f in pro_frames:
            f_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            results = pose_model.process(f_rgb)
            overlay = np.zeros_like(f)
            overlay = cv2.addWeighted(f, 0.6, overlay, 0.4, 0)
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                height, width, _ = overlay.shape
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    x1 = int(landmarks[start_idx].x * width)
                    y1 = int(landmarks[start_idx].y * height)
                    x2 = int(landmarks[end_idx].x * width)
                    y2 = int(landmarks[end_idx].y * height)
                    cv2.line(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
                    cv2.circle(overlay, (x1, y1), 6, (0, 255, 0), -1)
                    cv2.circle(overlay, (x2, y2), 6, (0, 255, 0), -1)
            pro_overlaid.append(overlay)

    # Encode pro frames as base64
    pro_frame_paths = []
    for frame in pro_overlaid:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        buf = BytesIO()
        pil_img.save(buf, format="PNG")
        base64_img = base64.b64encode(buf.getvalue()).decode('utf-8')
        pro_frame_paths.append(f"data:image/png;base64,{base64_img}")

    overlay_frame_paths = generate_overlay_frames(user_frames, pro_frames)

    return {
        "match": best_pro["name"],
        "similarity": similarity,
        "frames": frame_paths,
        "pro_frames": pro_frame_paths,
        "overlay_frames": overlay_frame_paths
    }

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    user_video = "MySwing.mp4"  # 🟢 Your input video
    db_file = "pro_pose_db.pkl"  # 🟢 Generated DB from earlier

    print("📦 Loading pro swing database...")
    with open(db_file, "rb") as f:
        pose_db = pickle.load(f)

    print("📹 Extracting your swing...")
    user_frames = extract_pose_frames(user_video)

    print("🤖 Finding closest pro swing...")
    best_pro, similarity = find_closest_pro(user_frames, pose_db)

    print(f"\n🎯 Closest Match: {best_pro['name']} ({similarity}%)")

    print("🎬 Loading pro swing...")
    pro_frames = extract_pose_frames(best_pro["video_path"])

    print("🪞 Launching side-by-side viewer...")
    user_frames_resampled = resample_vectors(user_frames, 60)
    pro_frames_resampled = resample_vectors(pro_frames, 60)

    frame_viewer(user_frames_resampled, pro_frames_resampled, similarity, best_pro["name"])