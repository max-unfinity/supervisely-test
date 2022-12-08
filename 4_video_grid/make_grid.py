from math import sqrt, ceil
from pathlib import Path
import subprocess
import os
import argparse


def check_better_fit(width, height, file):
    proc = subprocess.Popen('ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0'.split(' ')+[file], stdout=subprocess.PIPE)
    out = proc.stdout.readlines()[-1].decode().strip()
    w1,h1 = list(map(int, out.split('x')))
    return (width > height) != (w1 > h1)


def make_grid(path_to_dir, width=1920, height=1080, output='output.mp4', better_fit=True, ext='.*'):
    if ext[0] != '.':
        ext = '.'+ext
    list_video = sorted(list(map(str, Path(path_to_dir).glob(f'*{ext}'))))
    n_videos = len(list_video)
    try:
        if better_fit and check_better_fit(width, height, list_video[0]):
            width, height = height, width
            print('better_fit swaps width and height')
    except Exception as e:
        print(e)
        print("can't check for better fit, resuming...")
    
    grid_width = int(ceil(sqrt(n_videos)))
    grid_height = int(ceil(n_videos/grid_width))
    w,h = width//grid_width, height//grid_height
    input_commands = []
    input_setpts = f"color=s={width}x{height}:c=black [base];"
    fname = os.path.basename(list_video[0])
    input_overlays = f"[base][video0] overlay=shortest=1 [tmp0];[tmp0] drawtext=text='{fname}':x=({w}-text_w)/2:y=({h}-text_h):fontsize=32:fontcolor=white:box=1:boxcolor=black@0.5 [tmp0];"
    
    for index, path_video in enumerate(list_video):
        fname = os.path.basename(path_video)
        input_commands += ["-i", path_video]
        input_setpts += f"[{index}:v] setpts=PTS-STARTPTS, scale={w}x{h} [video{index}];"
        if index > 0 and index < len(list_video):
            x, y = w*(index%grid_width), h*(index//grid_width)
            input_overlays += f"[tmp{index-1}][video{index}] overlay=shortest=1:x={x}:y={y} [tmp{index}];"
            input_overlays += f"[tmp{index}] drawtext=text='{fname}':x={x}+({w}-text_w)/2:y={y+h}-text_h:fontsize=32:fontcolor=white:box=1:boxcolor=black@0.5 [tmp{index}];"
        if index == len(list_video) - 1:
            input_overlays = input_overlays[:-8]  # remove last unused "[tmp{index}];", otherwise ffmpeg will throw
    
    # complete_command = "ffmpeg" + input_videos + " -filter_complex \"" + input_setpts + input_overlays  + "\"" + " -c:v libx264 output.mp4"
    complete_command = ['ffmpeg'] + input_commands + ['-filter_complex', input_setpts + input_overlays, '-c:v', 'libx264', '-an', output]
    # print(complete_command)
    subprocess.run(complete_command)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_dir', type=str)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--output', type=str, default='output.mp4')
    parser.add_argument('--no_better_fit', action='store_true', help='don\'t try to set orientation (WxH or HxW) the same as input videos')
    parser.add_argument('--ext', type=str, default='.*', help='e.g. ".avi", ".mp4", ".*"')
    args = parser.parse_args()
    print(args)
    make_grid(args.path_to_dir, args.width, args.height, args.output, not args.no_better_fit, args.ext)
    