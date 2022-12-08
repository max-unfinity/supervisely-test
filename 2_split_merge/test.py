import numpy as np
import cv2
import os, shutil
import time

from split_merge import split, merge, merge_naive


N_TESTS = 10
N_THREADS = 6
MERGE_METHOD = 1
SEED = 43


def sample_params():
    img_size = np.random.randint(10,100, (2,))
    window_size = np.random.randint(10,100, (2,))
    stride_x = np.random.randint(1,window_size[1])
    stride_y = np.random.randint(1,window_size[0])
    strides = (stride_x, stride_y)
    padding = np.random.rand() < 0.5
    padding_all_sides = np.random.rand() < 0.5
    assert strides[0] <= window_size[1] and strides[1] <= window_size[0], 'step_size must be <= window_size'
    if np.random.rand() < 0.5:
        window_size = window_size / img_size
    img = np.random.randint(0,255, tuple(img_size)+(3,), dtype=np.uint8)
    return img_size, window_size, strides, padding, padding_all_sides, img


if __name__ == '__main__':
    # prepare dirs
    for i in range(N_TESTS):
        save_dir = 'test/splits_'+str(i)
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        # os.makedirs(save_dir, exist_ok=False)
        
    np.random.seed(SEED)
    start_time = time.time()
    for i in range(N_TESTS):
        save_dir = 'test/splits_'+str(i)
        
        img_size, window_size, strides, padding, padding_all_sides, img = sample_params()
        
        split(img, window_size, strides, save_dir, padding, padding_all_sides, N_THREADS)
        if MERGE_METHOD == 1:
            img_rec = merge(save_dir, n_threads=N_THREADS)
        else:
            img_rec = merge_naive(save_dir, n_threads=N_THREADS)
        cv2.imwrite(save_dir+'/rec.png', img_rec)
        
        diff = np.abs(img_rec.astype(float) - img.astype(float)).mean()
        assert diff == 0., str(i)
        
    print(time.time() - start_time, 'seconds')
