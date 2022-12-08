import os
import subprocess
import argparse
import random
import sys
sys.path.append('yolov7')

import pytube
from easydict import EasyDict

from yolov7.detect import detect


def download_frames(watch_url, fps=1.0, n_frames=50, save_dir='.'):
    '''Downloads with ffmpeg only needed `n_frames` frames (not the whole video).
    Start time is random.'''
    yt = pytube.YouTube(watch_url)
    stream = yt.streams.get_highest_resolution()
    url = stream.url
    img_dir = os.path.join(save_dir, yt.video_id)
    os.makedirs(img_dir, exist_ok=True)
    n_digits = len(str(n_frames))
    start_time = random.randint(0, yt.length)
    cmd = f'ffmpeg -y -ss {start_time} -i {url} -vf fps={fps},scale=640:-1 -frames:v {n_frames}'.split(' ') + [f'{img_dir}/img%0{n_digits}d.jpg']
    subprocess.run(cmd)
    return img_dir, yt.video_id
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('watch_url', type=str, help='e.g. https://www.youtube.com/watch?v=b41k2_MQNBk')
    parser.add_argument('--fps', type=float, default=1.0, help='lower fps gets more sparsed video frames')
    parser.add_argument('--n_frames', type=int, default=50)
    parser.add_argument('--save_frames_to', type=str, default='frames')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--weights', type=str, default='yolov7.pt')
    parser.add_argument('--save_predicts_to', type=str, default='predicts')
    args = parser.parse_args()
    
    img_dir, v_id = download_frames(args.watch_url, args.fps, args.n_frames, args.save_frames_to)
    
    # make default opt dict for yolo's detect.py
    opt = dict(agnostic_nms=False, augment=False, classes=None, conf_thres=0.25, device=args.device, exist_ok=False, img_size=640, iou_thres=0.45, name=v_id, no_trace=True, nosave=False, project=args.save_predicts_to, save_conf=False, save_txt=False, source=img_dir, update=False, view_img=False, weights=[args.weights])
    opt = EasyDict(opt)
    
    detect(opt)
    