import mediapipe as mp
import cv2
import numpy as np
from tqdm.auto import tqdm
import pydub
import json
import joblib

def write_landmarks_to_csv(landmarks, frame_number, csv_data):
    print(f"Landmark coordinates for frame {frame_number}:")
    for idx, landmark in enumerate(landmarks):
        print(f"{mp_pose.PoseLandmark(idx).name}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")
        csv_data.append([frame_number, mp_pose.PoseLandmark(idx).name, landmark.x, landmark.y, landmark.z])
    print("\n")

def get_mediapipe_landmarks(input_dir, output_dir):

    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_mesh = mp.solutions.face_mesh

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5)
    mesh = mp_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5)
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.2)

    cap = cv2.VideoCapture(input_dir)
    im_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    im_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Image w, h: ", im_w, im_h)
    print('Number of frames: ', n_frames)

    mp_results = []
    pbar = tqdm(total=n_frames)

    while cap.isOpened():
        pbar.update(1)
        success, image = cap.read()
        if not success:
            print("video end")
            break

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)
        results_hand = hands.process(image)
        results_pose = pose.process(image)
        results_mesh = mesh.process(image)
    
        mp_results.append((frame_no, dict(
            pose=results_pose.pose_landmarks,
            face=results_mesh.multi_face_landmarks,
            hands=results_hand.multi_hand_landmarks
        )))

    pbar.close()
    pose.close()
    mesh.close()
    hands.close()

    print(len(mp_results))
    joblib.dump(mp_results, output_dir)
    print('Done!')
    

if __name__ == '__main__':

    video_title = 'short_clip_01_crop'
    in_dir = f'./data/videos/{video_title}.mp4'
    out_dir = f'./data/result_mp/landmarks/mp_{video_title}.joblib'

    get_mediapipe_landmarks(in_dir, out_dir)
