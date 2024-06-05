import cv2
import subprocess

def get_cropped_video(video_title):

    input_dir = f'./data/videos/{video_title}.mp4'
    output_dir = f'./data/videos/{video_title}_crop.mp4'

    cap = cv2.VideoCapture(input_dir)
    im_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    im_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    print("Image w, h: ", im_w, im_h)
    print('Number of frames: ', n_frames)

    # ffmpeg -i before.mp4 -vf "crop=w:h:x:y" after.mp4

    try:
        subprocess.run(['ffmpeg', '-i', input_dir, '-vf', 'crop=640:720:640:0', output_dir])
        print('Successfully cropping video.')
    except Exception as e:
        print('Error:', e)
    
if __name__ == '__main__':

    video_title = 'short_clip_01'

    get_cropped_video(video_title)