import os
import cv2
import pickle
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose

# Pose vector extractor
def get_pose_vector(frame, pose_model):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose_model.process(frame_rgb)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        coords = np.array([[lm.x, lm.y] for lm in landmarks])
        center = (coords[11] + coords[12]) / 2  # center at shoulders
        coords -= center
        coords /= np.linalg.norm(coords) + 1e-6
        return coords.flatten()
    return None

# Extract and store all pro swing pose vectors
def generate_pro_pose_database(input_folder="proSwing", output_file="pro_pose_db.pkl"):
    pose_db = []
    video_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]

    with mp_pose.Pose(static_image_mode=True) as pose_model:
        for video_file in video_files:
            path = os.path.join(input_folder, video_file)
            print(f"Processing: {video_file}")
            cap = cv2.VideoCapture(path)

            frames = []
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                frames.append(frame)
            cap.release()

            pose_vectors = []
            for frame in frames:
                vec = get_pose_vector(frame, pose_model)
                if vec is not None:
                    pose_vectors.append(vec)

            pose_db.append({
                "name": video_file.replace("_Swing.mp4", "").replace(".mp4", ""),
                "video_path": path,
                "pose_vectors": pose_vectors
            })

    with open(output_file, "wb") as f:
        pickle.dump(pose_db, f)

    print(f"\n✅ Database saved to: {output_file}")

# Run script
if __name__ == "__main__":
    generate_pro_pose_database()