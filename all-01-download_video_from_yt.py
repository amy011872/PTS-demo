import os
import pytube
import argparse

def download_videos(link, save_path):
    try:
        my_video = pytube.YouTube(link, use_oauth=True, allow_oauth_cache=True)
        my_video = my_video.streams.get_highest_resolution()
        my_video.download(save_path)
        print('successfully downloaded')
    except Exception as e:
        print(link)
        print(e)

# argparse
def get_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--save_dir', type=str, default="./videos",
                        help='directory to save data')
    parser.add_argument('--target_url', type=str, required=True, 
                        help='target youtube url to download')
    args = parser.parse_args()
    return args
    


if __name__ == '__main__':
    
    args = get_arguments()

    save_dir = args.save_dir
    target_url = args.target_url
    
    download_videos(target_url, save_dir)