import json
import os
import copy
import random
import argparse
import numpy as np
import scipy.ndimage
from types import SimpleNamespace
NR = 10
SOI = 11
EOI = 12
SOO = 13
EOO = 14

chars = '.123456789'
# chars = '.oT+=|%~@#'
num2char = {}
for idx, c in enumerate(chars):
    num2char[idx] = c
num2char[NR] = "\\n"
num2char[SOI] = "<\\n"
num2char[EOI] = ">\\n"
num2char[SOO] = "[\\n"
num2char[EOO] = "]\\n"
# num2char[SOO] = "<\\n"
# num2char[EOO] = ">\\n"


parser = argparse.ArgumentParser()
parser.add_argument('--shuffle-train', action='store_true')
parser.add_argument('--repeat-frames', action='store_true')
parser.add_argument('--rotate', action='store_true')
parser.add_argument('--hue', action='store_true')
parser.add_argument('--flip', action='store_true')
parser.add_argument('--truncate', action='store_true')
parser.add_argument('--stretch', action='store_true')
parser.add_argument('--shear-roll', action='store_true')
parser.add_argument('--roll', action='store_true')
parser.add_argument('--back-tr', action='store_true')
parser.add_argument('--per-frame-random', action='store_true')
parser.add_argument('--times', type=int, default=0)
parser.add_argument('--name', type=str, default='arc_raw')
args = parser.parse_args()

def read_json(path_to_jsonfiles):
    alldicts = []
    # for idx, file in enumerate(os.listdir(path_to_jsonfiles), key = os.path.getsize):
    filenames = os.listdir(path_to_jsonfiles)
    files = sorted([os.path.join(path_to_jsonfiles, file) for file in filenames], key=os.path.getsize)
    for idx, file in enumerate(files):
        if 'evaluation' in file:
            # if idx < 1: continue
            if idx == 30: break
        full_filename = file
        with open(full_filename,'r') as fi:
            dict = json.load(fi)
            alldicts.append(dict)
    return alldicts

def augment(x, args, rands):
    def rotate(x, rotate_times):
        x = copy.deepcopy(x)
        for i in range(rotate_times):
            # import pdb;pdb.set_trace()
            x = [ list(a) for a in zip(*x[::-1])]
        return x

    def shift_hue(x, shift_val, order):
        # ok_nums = [a+1 for a  in range(9)] # ignore 0
        ok_nums = order
        # z = copy.deepcopy(x)
        def shift (y, shift_val):
            if y == 0:
                return y
            # new_hue_idx = (ok_nums.index(y)+shift_val)
            # y = ok_nums[new_hue_idx % len(ok_nums)]
            y = ok_nums[y-1]
            return y

        for idx, l in enumerate(x):
            x[idx] = [shift(num, shift_val) for num in l]
        # import pdb;pdb.set_trace()
        return x

    def stretch(x, strx, stry):
        x = np.array(x)
        temp_x = scipy.ndimage.zoom(x, (strx, stry))
        while temp_x.shape[0] > 30 or temp_x.shape[1] > 30:
            strx = max(1, strx-.1)
            stry = max(1, stry-.1)
            temp_x = scipy.ndimage.zoom(x, (strx, stry))
        return temp_x.tolist()

    def shear_roll(x, shx, shy):
        # import pdb;pdb.set_trace()
        x = np.array(x)
        for i in range(x.shape[0]):
            x[i,:] = np.roll(x[i,:], i+shx)
        for i in range(x.shape[1]):
            x[:,i] = np.roll(x[:,i], i+shy)
        return x.tolist()

    def roll(x, shx, shy):
        x = np.array(x)
        for i in range(x.shape[0]):
            x[i,:] = np.roll(x[i,:], shx)
        for i in range(x.shape[1]):
            x[:,i] = np.roll(x[:,i], shy)
        return x.tolist()

    def flip(x, do_flip):
        if do_flip:
            x = [i[::-1] for i in x[::-1]]
        return x
    if args.hue: x = shift_hue(x, rands.shift_val, rands.order)
    if args.flip: x = flip(x, rands.do_flip)
    if args.rotate: x = rotate(x, rands.rotate_times)
    if args.stretch: x = stretch(x, rands.stretch_x, rands.stretch_y)
    if args.roll: x = roll(x, rands.roll_x, rands.roll_y)
    if args.shear_roll: x = shear_roll(x, rands.shear_x, rands.shear_y)
    return x


def gen_seq(dirname, args, back_tr=False, pair=False, test=False):
    dicts = read_json(dirname)
    # mto = [d for d in train_dicts if len(d['test']) != 1]
    # import pdb; pdb.set_trace()
    srcs = []
    tgts = []
    for idx, d in enumerate(dicts):

        if not args.train_only:
            train = copy.deepcopy(d['train'])
            test = copy.deepcopy(d['test'])
        else:            
            train = copy.deepcopy(d['train'][:-1])
            test = copy.deepcopy(d['train'][-1:])
        if args.shuffle_train:
            train_test = copy.deepcopy(train) + copy.deepcopy(test)
            random.shuffle(train_test)
            train = train_test[:-1] # ok
            test = train_test[-1:]

        trunc_or_repeat = random.randint(0,2)
        if trunc_or_repeat == 0 and args.truncate:
            # print(len(train))
            until = random.randrange(len(train))
            if until == 0: 
                until = 1
            train = train[:until]

        elif trunc_or_repeat == 1 and args.repeat_frames:
            for idx, f in enumerate(train):
                if random.randint(0,1) == 1:
                    train.insert(idx, copy.deepcopy(f))
                    break # to limit too much frames



        def get_rands():
            rands = SimpleNamespace()
            rands.shift_val = random.randrange(9)+1
            rands.order = list(range(1,10))
            random.shuffle(rands.order)
            rands.rotate_times = random.randrange(2)+1
            rands.stretch_x = random.randrange(1,2)
            rands.stretch_y = random.randrange(1,2)
            rands.shear_x = random.randint(-1,1)
            rands.shear_y = random.randint(-1,1)
            rands.roll_x = random.randint(-3,3)
            rands.roll_y = random.randint(-3,3)
            rands.do_flip = random.randint(0,1) == 1
            return rands

        rands = get_rands()
        src = []
        for f in train:
            # if test: break

            finp = f['input'] if not back_tr else f['output']
            fout = f['output'] if not back_tr else f['input']
            if args.per_frame_random and random.randint(0,1) == 1:
                rands = get_rands()
            finp = augment(finp, args, rands)
            fout = augment(fout, args, rands)
            

            # src.extend([SOI])
            # for l in finp:
            #     src.extend(l + [NR,])
            # for l in fout:
            #     src.extend(l + [NR,])
            # src.extend([EOI])

            src.extend([SOI])
            for l in finp:
                src.extend(l + [NR,])
            src.extend([EOI])

            if pair:
                srcs.append(copy.deepcopy(src))
                src = []


            src.extend([SOO])
            for l in fout:
                src.extend(l + [NR,])
            src.extend([EOO])
            if pair:
                tgts.append(copy.deepcopy(src))
                src = []

        tgt = []
        for f in test:
            if 'output' not in f.keys(): 
                # test set, put dummy labels
                f['output'] = [[0,],[0,]]
            finp = f['input'] if not back_tr else f['output']
            fout = f['output'] if not back_tr else f['input']
            if args.per_frame_random and random.randint(0,1) == 1:
                rands = get_rands()
            finp = augment(finp, args, rands)
            fout = augment(fout, args, rands)

            
            temp_src = []
            temp_src.extend([SOI])
            for l in finp:
                temp_src.extend(l + [NR,])
            temp_src.extend([EOI])
            if pair:
                srcs.append(copy.deepcopy(temp_src))
            else:
                srcs.append(copy.deepcopy(src + temp_src)) # we need to change src if not end seq

            # tgt = []
            # src.extend([SOO])
            # for l in fout:
            #     tgt.extend(l + [NR,])
            # tgts.append(tgt)
            # src.extend([EOO])
            tgt = []
            tgt.extend([SOO])  
            for l in fout:
                tgt.extend(l + [NR,])
            tgt.extend([EOO])
            tgts.append(tgt)
            # if len(test) >1:import pdb;pdb.set_trace()
    return srcs, tgts

    # if not swap_train_test:
    #     return srcs, tgts
    # else:
    #     return tgts, srcs

def write_dataset(srcs, tgts, filename):
    with open(filename+".src", "w") as src_text:
        for src in srcs:
            # src = [str(i) for i in src]back_tr
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
ts = []
tt = []
def append_s_t(tt, ts, folders, args, back_tr, pair):
    for folder in folders:
        for i in range(args.times):
            s, t = gen_seq(folder, to_args, back_tr, pair)
            ts += s
            tt += t
def nohue_append(tt, ts, folders, args):
    nohue_args = copy.deepcopy(args)
    nohue_args.hue = False
    nohue_args.times = 1
    append_s_t(tt, ts, folders, nohue_args, back_tr=False, pair=False)
    append_s_t(tt, ts, folders, nohue_args, back_tr=True, pair=False)
    # append_s_t(tt, ts, folders, nohue_args, back_tr=False, pair=True)
    # append_s_t(tt, ts, folders, nohue_args, back_tr=True, pair=True)
    return nohue_args

def nopfr_append(tt, ts, folders, args):
    nopfr_args = copy.deepcopy(args)
    nopfr_args.per_frame_random = False
    append_s_t(tt, ts, folders, nopfr_args, back_tr=False, pair=False)
    append_s_t(tt, ts, folders, nopfr_args, back_tr=True, pair=False)
    # append_s_t(tt, ts, folders, nopfr_args, back_tr=False, pair=True)
    # append_s_t(tt, ts, folders, nopfr_args, back_tr=True, pair=True)
    return nopfr_args


default_args = SimpleNamespace()
for key in vars(args):
    setattr(default_args, key, parser.get_default(key))

folders = ['evaluation']

# default
default_args.train_only = True
append_s_t(tt, ts, folders, default_args, back_tr=False, pair=False)
append_s_t(tt, ts, folders, default_args, back_tr=True, pair=False)
# append_s_t(tt, ts, folders, default_args, back_tr=False, pair=True)
# append_s_t(tt, ts, folders, default_args, back_tr=True, pair=True)
default_args.train_only = False
#
# current args
to_args = copy.deepcopy(args)
to_args.train_only = True
append_s_t(tt, ts, folders, to_args, back_tr=False, pair=False)
append_s_t(tt, ts, folders, to_args, back_tr=True, pair=False)
# append_s_t(tt, ts, folders, to_args, back_tr=False, pair=True)
# append_s_t(tt, ts, folders, to_args, back_tr=True, pair=True)

# nohue
if to_args.hue == True:
    nohue_append(tt, ts, folders, to_args)
if to_args.per_frame_random == True:
    nopfr_args = nopfr_append(tt, ts, folders, to_args)
    if to_args.hue == True: # note that args passed here perframeandom false
        nohue_append(tt, ts, folders, nopfr_args)
    
# valid set
# vs, vt = gen_seq('evaluation', default_args)
# vs, vt = gen_seq('evaluation', default_args, pair=True, test=True)
vs, vt = gen_seq('evaluation', default_args, test=True)

# finally write to files
write_dataset(ts, tt, args.name+'/train')
write_dataset(vs, vt, args.name+'/valid')

# # easiest dataset, same train-val
# vs, vt = gen_seq('evaluation', empty_args)
# write_dataset(vs, vt, args.name+'/valid')
# write_dataset(vs, vt, args.name+'/train')

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
