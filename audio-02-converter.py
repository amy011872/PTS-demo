import os
import subprocess
import re
from tqdm import tqdm

'''For audio processing, first convert .mp4 to .wav'''

def convert2wav(video_title, in_path, out_path):
    try:
        subprocess.run(['ffmpeg', '-i', in_path, '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le', '-vn',  out_path])
        print('Successfully convert', video_title, 'to .wav')
    except Exception as e:
        print('Error:', e)
        print(video_title)

if __name__ == '__main__':

    videos = os.listdir('./data/videos')
    for video in tqdm(videos):
        video_title = video
        video_path = os.path.join('./data/videos', video)
        wav_path = os.path.join('./data/wavs/', re.sub('.mp4', '.wav', video))

        if not os.path.exists(wav_path):
            convert2wav(video_title, video_path, wav_path)

    print('Done!')