import numpy as np
from pathlib import Path
import json
from PIL import Image, ImageDraw
import os
# from matplotlib import pyplot as plt

from imgviz.imgviz.instances import instances2rgb
from imgviz.imgviz import label as label_module


def visualize(img, bboxes, masks, labels, probs, colors=None, mask_alpha=0.5, font_scale=1.5):
    captions = [f'{l} {p:.2f}' for l,p in zip(labels, probs)]
    res = instances2rgb(img, bboxes, masks, captions=captions, colors=colors, alpha=mask_alpha, font_scale=font_scale)
    return res


def polygons2mask(polygons, wh):
    mask = Image.new('L', wh)
    mask_d = ImageDraw.Draw(mask)
    polygons = [tuple(tuple(x[i:i+2]) for i in range(0,len(x),2)) for x in polygons]
    for p in polygons:
        mask_d.polygon(p, fill=1)
    mask = np.array(mask)
    return mask


def preprocess_coco(annotations, label2name):
    # extract masks, bboxes and labels
    annotations = map(lambda x: list(x.values()), annotations)
    annotations = filter(lambda x: not x[2], annotations)
    seg,_,_,_,bboxes,labels,_ = list(zip(*annotations))
    bboxes = np.array(bboxes)
    bboxes[:,2:] += bboxes[:,:2]
    
    # sorting by area (bigger objects will drawn first)
    arg_sort = (bboxes[:,2:]-bboxes[:,:2]).prod(1).argsort()[::-1]
    bboxes = bboxes[arg_sort]
    seg,labels = list(zip(*[[seg[i],labels[i]] for i in arg_sort]))
    
    labels = [label2name[str(l)] for l in labels]
    masks = [polygons2mask(seg_i, img_pil.size)==1 for seg_i in seg]
    return bboxes, masks, labels


if __name__ == '__main__':
    with open('data/anns.json','r') as f:
        anns = json.load(f)
    with open('data/label2name.json','r') as f:
        label2name = json.load(f)
        
    os.makedirs('result', exist_ok=True)
    for ann, img_name in anns:
        # ann, img_name = anns[1]
        img_pil = Image.open('data/'+img_name)
        img = np.array(img_pil)
        bboxes, masks, labels = preprocess_coco(ann, label2name)
        
        probs = np.random.rand(len(labels))**(1/8)
        colors = label_module.label_colormap()[1:]
        
        res = visualize(img, bboxes, masks, labels, probs, colors)
        
        res2 = Image.fromarray(res)
        res2.save(f'result/{img_name}.png')
        # res2.show()
        # break
        # plt.imshow(res)
        # plt.axis('off')
        # plt.tight_layout()
        
        
