import joblib
import json
from tqdm import tqdm
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions import face_mesh_connections
from typing import Mapping, Tuple
from mediapipe.python.solutions.pose import PoseLandmark
from PIL import Image, ImageDraw

def draw_skeleton_video(video_title, frame_file_name):

    mp_results = joblib.load(f'./data/result_mp/landmarks/{video_title}.joblib')
    print(len(mp_results))

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose
    mp_mesh = mp.solutions.face_mesh

    # colors
    _LIGHTGREEN = (247, 255, 247)
    _DARKGREEN = (26,83,92)
    _BUTTONGREEN = (4,156,132)
    _PEACH = (255, 107, 107)
    _YELLOW = (255, 230, 109)

    hand_style = mp_drawing_styles.get_default_hand_landmarks_style()
    for k, v in hand_style.items():
        v.color = _DARKGREEN
        v.circle_radius=1

    hand_conn_style = mp_drawing_styles.get_default_hand_connections_style()
    for k, v in hand_conn_style.items():
        v.color = _DARKGREEN
        v.circle_radius=1
        v.thickness=1

    iris_style = mp_drawing_styles.get_default_face_mesh_iris_connections_style()
    for k, v in iris_style.items():
        if 48 in v.color:
            v.color=_DARKGREEN

    hands_mask = np.array([x[1]["hands"] is not None for x in mp_results])
    to_draw_hands = hands_mask.copy()
    for i, val in enumerate(hands_mask):
        s = max(i-15, 0)
        e = min(i+15, len(hands_mask))
        to_draw_hands[i] = np.all(hands_mask[s:e])

    '''自定義_FACEMESH_CONTOURS_CONNECTION_STYLE'''
    _THICKNESS_TESSELATION = 1
    _THICKNESS_CONTOURS = 1
    _THICKNESS_FACE_OVAL = 2

    _FACEMESH_CONTOURS_CONNECTION_STYLE = {
        face_mesh_connections.FACEMESH_LIPS:
            DrawingSpec(color=_PEACH, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_LEFT_EYE:
            DrawingSpec(color=_DARKGREEN, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_LEFT_EYEBROW:
            DrawingSpec(color=_DARKGREEN, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_RIGHT_EYE:
            DrawingSpec(color=_DARKGREEN, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_RIGHT_EYEBROW:
            DrawingSpec(color=_DARKGREEN, thickness=_THICKNESS_CONTOURS),
        face_mesh_connections.FACEMESH_FACE_OVAL:
            DrawingSpec(color=_YELLOW, thickness=_THICKNESS_FACE_OVAL)
    }

    def getMyFaceMeshContours():
        face_mesh_contours_connection_style = {}
        for k, v in _FACEMESH_CONTOURS_CONNECTION_STYLE.items():
            for connection in k:
                face_mesh_contours_connection_style[connection] = v
        return face_mesh_contours_connection_style

    '''自定義 POSE_LANDMARKS'''
    _THICKNESS_POSE_LANDMARKS = 2
    _POSE_LANDMARKS_LEFT = frozenset([
        PoseLandmark.LEFT_EYE_INNER, PoseLandmark.LEFT_EYE,
        PoseLandmark.LEFT_EYE_OUTER, PoseLandmark.LEFT_EAR, PoseLandmark.MOUTH_LEFT,
        PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW,
        PoseLandmark.LEFT_WRIST, PoseLandmark.LEFT_PINKY, PoseLandmark.LEFT_INDEX,
        PoseLandmark.LEFT_THUMB, PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE,
        PoseLandmark.LEFT_ANKLE, PoseLandmark.LEFT_HEEL,
        PoseLandmark.LEFT_FOOT_INDEX
    ])

    _POSE_LANDMARKS_RIGHT = frozenset([
        PoseLandmark.RIGHT_EYE_INNER, PoseLandmark.RIGHT_EYE,
        PoseLandmark.RIGHT_EYE_OUTER, PoseLandmark.RIGHT_EAR,
        PoseLandmark.MOUTH_RIGHT, PoseLandmark.RIGHT_SHOULDER,
        PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST,
        PoseLandmark.RIGHT_PINKY, PoseLandmark.RIGHT_INDEX,
        PoseLandmark.RIGHT_THUMB, PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE,
        PoseLandmark.RIGHT_ANKLE, PoseLandmark.RIGHT_HEEL,
        PoseLandmark.RIGHT_FOOT_INDEX
    ])

    def getMyPoseLandmarkStyle():
        """Returns the default pose landmarks drawing style.
        Returns:
            A mapping from each pose landmark to its default drawing spec.
        """
        pose_landmark_style = {}
        left_spec = DrawingSpec(
            color=_BUTTONGREEN, thickness=_THICKNESS_POSE_LANDMARKS)
        right_spec = DrawingSpec(
            color=_BUTTONGREEN, thickness=_THICKNESS_POSE_LANDMARKS)
        
        for landmark in _POSE_LANDMARKS_LEFT:
            pose_landmark_style[landmark] = left_spec
        for landmark in _POSE_LANDMARKS_RIGHT:
            pose_landmark_style[landmark] = right_spec

        pose_landmark_style[PoseLandmark.NOSE] = DrawingSpec(
            color=_YELLOW, thickness=_THICKNESS_POSE_LANDMARKS)
        return pose_landmark_style

    # n_hand, n_face, n_pose = 0, 0, 0
    # for frame_idx, results in tqdm(mp_results):
    #     pose_landmarks = results["pose"]
    #     results_hands = results["hands"]
    #     face_landmarks = results["face"]
    #     if face_landmarks:
    #         n_face += 1
    #     if results_hands:
    #         n_hand += 1
    #     if pose_landmarks:
    #         n_pose += 1
    # print(n_hand, n_face, n_pose) ## 854 22 1106


    LEFT_WRIST = mp_pose.PoseLandmark.LEFT_WRIST
    RIGHT_WRIST = mp_pose.PoseLandmark.RIGHT_WRIST
    buf = [{} for _ in mp_pose.PoseLandmark]
    for frame_idx, results in tqdm(mp_results):
        pose_landmarks = results["pose"]
        results_hands = results["hands"]
        face_landmarks = results["face"]
        annotated_image = np.ones((480,640,3), dtype=np.uint8)
        
        if pose_landmarks:
            for i, coord in enumerate(pose_landmarks.landmark):    
                coord.x = coord.x * .2 + buf[i].get("x", coord.x) * .8
                coord.y = coord.y * .2 + buf[i].get("y", coord.y) * .8        
                buf[i]["x"] = coord.x
                buf[i]["y"] = coord.y
                if 11 <= i <= 16 or 23<=i<=24:
                    coord.visibility = .99
                else:
                    coord.visibility = 0.
            
        if results_hands and to_draw_hands[int(frame_idx-1)]:
                for hand_landmarks in results_hands:
                    mp_drawing.draw_landmarks(
                        annotated_image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        hand_style,
                        hand_conn_style)
                    wrist_landmark = hand_landmarks.landmark[0]
                    if wrist_landmark.x < 0.5:
                        pose_landmarks.landmark[RIGHT_WRIST].x = wrist_landmark.x
                        pose_landmarks.landmark[RIGHT_WRIST].y = wrist_landmark.y
                    else:
                        pose_landmarks.landmark[LEFT_WRIST].x = wrist_landmark.x
                        pose_landmarks.landmark[LEFT_WRIST].y = wrist_landmark.y
            
        # img = Image.fromarray(annotated_image)

        if pose_landmarks:    
            ### mp_pose.POSE_CONNECTIONS
            mp_drawing.draw_landmarks(
                    annotated_image,
                    pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=getMyPoseLandmarkStyle(),
                    connection_drawing_spec= DrawingSpec(color=_LIGHTGREEN, thickness=2))
            
        # img = Image.fromarray(annotated_image)
        
        if face_landmarks:
            # print(face_landmarks[0])
            ### mp_mesh.FACEMESH_TESSELATION
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks[0],
                connections=mp_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec= DrawingSpec(color=_LIGHTGREEN, thickness=1))
                
            ### mp_mesh.FACEMESH_CONTOURS
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks[0],
                connections=mp_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=getMyFaceMeshContours())
                
            ### mp_mesh.FACEMESH_IRISES
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks[0],
                connections=mp_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=iris_style)      
        
        img = Image.fromarray(annotated_image)

        try:
            img.save(f"./data/result_mp/{frame_file_name}/frames_{int(frame_idx):03d}.png")     
            print(f'frames_{int(frame_idx):03d}.png saved')
        except Exception as e:
            print('Exception:', e)


if __name__ == '__main__':

    video_title = 'mp_short_clip_01_crop'
    frame_file_name = 'frames_short_clip_01_crop'

    draw_skeleton_video(video_title, frame_file_name)


 # After drawing frames, concatenate it into video 
 # >> cd ./data/result_mp/frames_short_clip_01/ && ffmpeg -i frames_%03d.png -framerate 25 ../skl_video/skl_short_clip_01.mp4
