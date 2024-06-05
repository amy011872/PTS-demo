import whisper
from tqdm import tqdm
import json
import re
import os

def use_whisper(video_title, wavfile):

    model = whisper.load_model("medium.en")
    wav_path = os.path.join('./data/wavs/', re.sub('.mp4', '.wav', wavfile))
    result = model.transcribe(wav_path, fp16=False)

    return result['segments']

def get_transcription_with_whisper(input_file, output_file):

    # input: video title lists
    with open(input_file, 'r', encoding='utf8') as f:
        video_meta = json.load(f)
    
    # output: transcription result
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf8') as f:
            whisper_dict = json.load(f)
    else:
        print('Error reading previous files')
    
    for video_title in tqdm(video_meta):
        wav_path = os.path.join('./data/wavs/', video_title)
        whisper_dict[video_title] = use_whisper(video_title, wav_path)

        # save transcription in each round
        with open(output_file, "w") as f:
            json.dump(whisper_dict, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':

    input_file = './data/video_meta.json'
    output_file = './data/result_whisper/result_whisper.json'

    get_transcription_with_whisper(input_file, output_file)
    print('Done!')