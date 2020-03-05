import json
import os
import copy
import random
import argparse
import cv2
import numpy as np
from torchvision.utils import save_image
import torch
from os.path import join
import math
from tqdm import tqdm
BCOL = 11
#COLORS = [math.ceil(i/11*255) for i in range(11)]
#NUMS = [math.floor(i/255*11) for i in COLORS]
parser = argparse.ArgumentParser()
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--train-only', action='store_true')
parser.add_argument('--hue', action='store_true')
parser.add_argument('--per-frame-random', action='store_true')
parser.add_argument('--times', type=int, default=1)
parser.add_argument('--name', type=str, default='arc_imgs')
args = parser.parse_args()


def read_json(path_to_jsonfiles):
    alldicts = []
    for file in os.listdir(path_to_jsonfiles):
        full_filename = "%s/%s" % (path_to_jsonfiles, file)
        with open(full_filename,'r') as fi:
            d = json.load(fi)
            d['name'] = os.path.splitext(file)[0]
            alldicts.append(d)

    return alldicts

def augment(img, args):
    img = img.reshape(*img.shape, 1)
    canvas = np.ones((30, 30, 1))
    canvas[:,:,0] = np.full((30, 30), BCOL)
    offset = np.zeros(2)
    # print(img.shape)
    canvas[int(offset[0]):img.shape[0], 
            int(offset[1]):img.shape[1],:] = img
    canvas = canvas / BCOL * 255
    canvas = np.floor(canvas)
    return canvas

def gen_imgs(srcdir, tgtdir, args):
    dicts = read_json(srcdir)
    # mto = [d for d in train_dicts if len(d['test']) != 1]
    # import pdb; pdb.set_trace()
    print(f"Reading {srcdir}")
    for i, d in tqdm(enumerate(dicts)):
        dirname = join(tgtdir, d['name'])
        os.makedirs(dirname,exist_ok=True)
        if not args.train_only:
            train = copy.deepcopy(d['train'])
            test = copy.deepcopy(d['test'])
        else:
            train = copy.deepcopy(d['train'][:-1])
            test = copy.deepcopy(d['train'][-1:])

        for j, f in enumerate(train):
            for k in range(args.times):
                fpi = np.array(f['input']) # frame pre-input
                fpo = np.array(f['output'])
                fi = augment(fpi, args)
                fo = augment(fpo, args)
                cv2.imwrite(join(dirname, f'train_{j}_inp_{k}.png'), fi)
                cv2.imwrite(join(dirname, f'train_{j}_out_{k}.png'), fo)

        for j, f in enumerate(test):
            if 'output' not in f.keys(): 
                f['output'] = [[0,]] # dummy output
            for k in range(args.times):
                fpi = np.array(f['input']) # frame pre-input
                fpo = np.array(f['output'])
                fi = augment(fpi, args)
                fo = augment(fpo, args)
                cv2.imwrite(join(dirname, f'test_{j}_inp_{k}.png'), fi)
                cv2.imwrite(join(dirname, f'test_{j}_out_{k}.png'), fo)
            

def write_dataset(srcs, tgts, filename):
    with open(filename+".src", "w") as src_text:
        for src in srcs:
            # src = [str(i) for i in src]
            src = [num2char[i] for i in src]
            src_text.write(' '.join(src))
            src_text.write("\n")
    with open(filename+".tgt", "w") as tgt_text:
        for tgt in tgts:
            # tgt = [str(i) for i in tgt]
            tgt = [num2char[i] for i in tgt]
            tgt_text.write(' '.join(tgt))
            tgt_text.write("\n")

os.makedirs(args.name, exist_ok=True)
gen_imgs('training', join(args.name,'train'), args)


'''
training
9 2
11 2
47 2
59 2
66 2
71 2
124 2
130 2
141 2
170 2
185 2
292 2
308 3
380 3
'''
'''
dev
57 2
60 2
66 2
74 2
138 2
143 2
165 2
181 2
184 2
187 2
216 2
243 2
263 2
311 2
343 2
350 2
372 2
386 2
394 2
'''
'''
test
17 2
52 2
86 2
97 2
'''
'''
train_dicts[0]['train'][0]['output']
'test'
 train_dicts[0]
{'test': [{'input': [[4, 0, 0, 0], [0, 0, 0, 4], [4, 4, 0, 0]], 
'output': [[4, 0, 0, 0, 0, 0, 0, 4], [0, 0, 0, 4, 4, 0, 0, 0],
 [4, 4, 0, 0, 0, 0, 4, 4], [4, 4, 0, 0, 0, 0, 4, 4],
  [0, 0, 0, 4, 4, 0, 0, 0], [4, 0, 0, 0, 0, 0, 0, 4]]}], 
  'train': [{'input': [[0, 0, 8, 0], [0, 8, 0, 8], [0, 0, 8, 0]], 
  'output': [[0, 0, 8, 0, 0, 8, 0, 0], [0, 8, 0, 8, 8, 0, 8, 0], 
  [0, 0, 8, 0, 0, 8, 0, 0], [0, 0, 8, 0, 0, 8, 0, 0],
   [0, 8, 0, 8, 8, 0, 8, 0], [0, 0, 8, 0, 0, 8, 0, 0]]}, 
   {'input': [[0, 0, 3, 3], [0, 3, 0, 3], [3, 3, 3, 0]], 
   'output': [[0, 0, 3, 3, 3, 3, 0, 0], [0, 3, 0, 3, 3, 0, 3, 0], [3, 3, 3, 0, 0, 3, 3, 3], [3, 3, 3, 0, 0, 3, 3, 3], [0, 3, 0, 3, 3, 0, 3, 0], [0, 0, 3, 3, 3, 3, 0, 0]]}, {'input': [[3, 3, 3, 3], [3, 0, 0, 0], [3, 0, 0, 0]], 'output': [[3, 3, 3, 3, 3, 3, 3, 3], [3, 0, 0, 0, 0, 0, 0, 3], [3, 0, 0, 0, 0, 0, 0, 3], [3, 0, 0, 0, 0, 0, 0, 3], [3, 0, 0, 0, 0, 0, 0, 3], [3, 3, 3, 3, 3, 3, 3, 3]]}]}
'''
