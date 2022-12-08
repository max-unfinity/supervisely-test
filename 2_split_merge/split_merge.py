import numpy as np
import cv2
from pathlib import Path
import re
import os


def split(src_img, window_size, strides, save_dir, padding=False, padding_all_sides=True, n_threads=4):
    """
    Parameters
    ----------
    src_img : np.array(HxWx3)
    window_size : tuple(h,w)
    strides : tuple(x,y)
        strides is step size in x,y
    save_dir : str
        directory where to save croped images
    padding : bool, optional
    padding_all_sides : bool, optional
        if True, the image will be in center, padding will cover all 4 borders,
        otherwise the image will be at left top.
    n_threads : int, optional
        if n_threads > 0, I/O operations will be threaded. The default is 4.
    """
    assert src_img.ndim > 1
    if src_img.ndim == 2:
        src_img[...,None].repeat(3, axis=2)
    
    window_size = list(window_size)
    if isinstance(window_size[0], float):
        window_size[0] = int(src_img.shape[0]*window_size[0])
    if isinstance(window_size[1], float):
        window_size[1] = int(src_img.shape[1]*window_size[1])
        
    assert window_size[0] >= strides[1] and window_size[1] >= strides[0], 'strides must be <= window_size'
        
    yy = np.arange(0, src_img.shape[0], strides[1])
    xx = np.arange(0, src_img.shape[1], strides[0])
    starts = np.stack(np.meshgrid(yy,xx,indexing='xy')).T
    ends = starts + np.array(window_size)

    if padding:
        img, pad = pad_image(src_img, window_size, strides, padding_all_sides=padding_all_sides)
    else:
        img, pad = src_img, (0,0,0,0)

    imgs = [img[aa[0]:bb[0],aa[1]:bb[1]] for a,b in zip(starts, ends) for aa,bb in zip(a,b)]

    # skips are for smart merging: we won't load all the pieces of image. This gives us better performance in merge
    skip_y = window_size[0]//strides[1]
    take_y = skip_y*strides[1]
    skip_x = window_size[1]//strides[0]
    take_x = skip_x*strides[0]

    skips = np.zeros(starts.shape[:2], dtype=int)
    skips[::skip_y, ::skip_x] = 1
    skips[-1,::skip_x] = 1
    skips[::skip_y, -1] = 1
    skips[-1,-1] = 1
    skips = skips.flatten()

    starts_flat = starts.reshape(-1, 2)
    n_digits = len(str(len(imgs)))+1
    coords = np.concatenate([starts_flat, starts_flat + np.array([take_y,take_x])], -1)
    coords_str = ["_".join(map(str, tuple(x))) for x in coords]
    img_size_str = f'{img.shape[0]}_{img.shape[1]}'
    pad_str = '_'.join(map(str, pad))

    os.makedirs(save_dir, exist_ok=False)

    def write_one(args):
        i,(img,coord,skip) = args
        save_name = f'{save_dir}/{i:0{n_digits}}_{img_size_str}_{coord}_{pad_str}_{skip}.png'
        cv2.imwrite(save_name, img[...,::])

    x = list(enumerate(zip(imgs,coords_str,skips)))
    if n_threads:
        multi_thread(write_one, x, n_threads=n_threads)
    else:
        list(map(write_one, x))
    
    
def merge(save_dir, ext='*', n_threads=4):
    '''
    Parameters
    ----------
    save_dir : str
        directory where the croped images were saved.
    ext : str, optional
        image extension, e.g. 'jpg', 'png'. Default is any extension.
    n_threads : int, optional
        if n_threads > 0, I/O operations will be threaded. The default is 4.

    Returns
    -------
    reconstruced image: np.array(HxWx3)
    '''
    files = sorted(list(map(str, Path(save_dir).glob(f'*.{ext}'))))
    fnames = list(map(lambda f: os.path.basename(f).split('.')[0], files))

    def read_params_from_filename(fname):
        return list(map(int, re.findall('\d+', fname)))

    info = read_params_from_filename(fnames[0])
    img_size = info[1:3]
    pad_left, pad_top, pad_right, pad_bottom = info[7:11]

    params = [[f]+read_params_from_filename(fname) for f,fname in zip(files,fnames)]
    params = [x for x in params if x[-1]]

    img_rec = np.zeros(img_size+[3], dtype=np.uint8)
    
    def load_one(x):
        return cv2.imread(x[0])

    if n_threads:
        loaded_imgs = multi_thread(load_one, params, n_threads=n_threads)
    else:
        loaded_imgs = list(map(load_one, params))
        
    for loaded, t in zip(loaded_imgs, params):
        f,i,_,_,y,x,y2,x2,_,_,_,_,take = t
        y2,x2 = loaded.shape[:2]
        y2,x2 = y+y2, x+x2
        img_rec[y:y2,x:x2] = loaded #[:y2-y,:x2-x,::]
        
    pad_bottom = img_rec.shape[0] - pad_bottom
    pad_right = img_rec.shape[1] - pad_right
    img_rec = img_rec[pad_top:pad_bottom,pad_left:pad_right]
    return img_rec


def merge_naive(save_dir, n_threads=4):
    '''
    Parameters
    ----------
    save_dir : str
        directory where the croped images were saved.
    n_threads : int, optional
        if n_threads > 0, I/O operations will be threaded. The default is 4.

    Returns
    -------
    reconstruced image: np.array(HxWx3)
    '''
    files = sorted(list(map(str, Path(save_dir).glob('*.*'))))
    fnames = list(map(lambda f: os.path.basename(f).split('.')[0], files))

    def read_params_from_filename(fname):
        return list(map(int, re.findall('\d+', fname)))

    info = read_params_from_filename(fnames[0])
    img_size = info[1:3]
    pad_left, pad_top, pad_right, pad_bottom = info[7:11]

    params = [[f]+read_params_from_filename(fname) for f,fname in zip(files,fnames)]

    img_rec = np.zeros(img_size+[3], dtype=np.uint8)
    def load_one(x):
        return cv2.imread(x[0])

    if n_threads:
        loaded_imgs = multi_thread(load_one, params, n_threads=n_threads)
    else:
        loaded_imgs = list(map(load_one, params))
        
    for loaded, t in zip(loaded_imgs, params):
        f,i,_,_,y,x,y2,x2,_,_,_,_,take = t
        y2,x2 = loaded.shape[:2]
        y2,x2 = y+y2, x+x2
        img_rec[y:y2,x:x2] = loaded #[:y2-y,:x2-x,::]
    pad_bottom = img_rec.shape[0] - pad_bottom
    pad_right = img_rec.shape[1] - pad_right
    img_rec = img_rec[pad_top:pad_bottom,pad_left:pad_right]
    return img_rec


def pad_image(img, window_size, strides, padding_all_sides=True):
    img_size_xy = np.array(img.shape[1::-1])
    win_size_xy = np.array(window_size[::-1])
    strides_xy = np.array(strides)

    pad = strides_xy*np.ceil(img_size_xy/strides_xy-1) + win_size_xy - img_size_xy

    if padding_all_sides:
        pad_left, pad_top, pad_right, pad_bottom = np.concatenate([np.ceil(pad/2), pad//2]).astype(int)
    else:
        pad_left, pad_top, (pad_right, pad_bottom) = 0, 0, pad.astype(int)

    img = np.pad(img, ((pad_top,pad_bottom),(pad_left,pad_right),(0,0)))
    return img, (pad_left, pad_top, pad_right, pad_bottom)


def multi_thread(fn, x, n_threads=4):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(n_threads) as pool:
        return list(pool.map(fn, x))
        # return list(tqdm(pool.map(fn, x), position=0, total=len(x)))
    