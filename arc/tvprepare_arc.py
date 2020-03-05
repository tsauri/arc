import json
import os
import copy
import random
import argparse
NR = 10
SOI = 11
EOI = 12
SOO = 13
EOO = 14

chars = '.oT+=|%~@#'
num2char = {}
for idx, c in enumerate(chars):
    num2char[idx] = c
num2char[NR] = "\\n"
num2char[SOI] = "<\\n"
num2char[EOI] = ">\\n"
num2char[SOO] = "[\\n"
num2char[EOO] = "]\\n"


parser = argparse.ArgumentParser()
parser.add_argument('--shuffle-train', action='store_true')
parser.add_argument('--shuffle-row', action='store_true')
parser.add_argument('--random-rotate', action='store_true')
parser.add_argument('--random-hue', action='store_true')
parser.add_argument('--random-truncate', action='store_true')
parser.add_argument('--per-frame-random', action='store_true')
parser.add_argument('--times', type=int, default=1)
parser.add_argument('--name', type=str, default='arc_raw')
args = parser.parse_args()

def read_json(path_to_jsonfiles):
    alldicts = []
    for file in os.listdir(path_to_jsonfiles):
        full_filename = "%s/%s" % (path_to_jsonfiles, file)
        with open(full_filename,'r') as fi:
            dict = json.load(fi)
            alldicts.append(dict)
    return alldicts

def gen_seq(dirname, args):
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
        if args.random_truncate:
            # print(len(train))
            until = random.randrange(len(train)) # ok
            if until == 0: until = 1
            train = train[:until]

        def rotate(x, rotate_times):
            x = copy.deepcopy(x)
            for i in range(rotate_times):
                # import pdb;pdb.set_trace()
                x = [ list(a) for a in zip(*x[::-1])]
            return x

        def shift_hue(x, shift_va):
            ok_nums = [a+1 for a  in range(9)] # ignore 0
            def shift(y, shift_val):
                if y == 0:
                    return y
                new_hue_idx = (ok_nums.index(y)+shift_val)
                y = ok_nums[new_hue_idx % len(ok_nums)]
                return y

            for l in x:
                l = [shift(num, shift_val) for num in l]
            return x

        # fix random
        shift_val = random.randrange(9)+1
        rotate_times = random.randrange(2)+1
        src = []
        for f in train:
            if args.per_frame_random:
                shift_val = random.randrange(9)+1
                rotate_times = random.randrange(2)+1
            if args.random_hue:
                f['input'] = shift_hue(f['input'], shift_val)
                f['output'] = shift_hue(f['output'], shift_val)
            if args.random_rotate:
                # import pdb;pdb.set_trace()
                f['input'] = rotate(f['input'], rotate_times)
                f['output'] = rotate(f['output'], rotate_times)

            src.extend([SOI])
            for l in f['input']:
                src.extend(l + [NR,])
            src.extend([EOI])
            src.extend([SOO])
            for l in f['output']:
                src.extend(l + [NR,])
            src.extend([EOO])

        tgt = []
        for f in test:
            if args.per_frame_random:
                shift_val = random.randrange(9)+1
                rotate_times = random.randrange(2)+1
            if 'output' not in f.keys(): 
                # test set, put dummy labels
                f['output'] = [[0,]]
            if args.random_hue:
                f['input'] = shift_hue(f['input'], shift_val)
                f['output'] = shift_hue(f['output'], shift_val)
            if args.random_rotate:
                f['input'] = rotate(f['input'], rotate_times)
                f['output'] = rotate(f['output'], rotate_times)
            temp_src = []
            temp_src.extend([SOI])
            for l in f['input']:
                temp_src.extend(l + [NR,])
            temp_src.extend([EOI])
            srcs.append(copy.deepcopy(src + temp_src)) # we need to change src if not end seq

            tgt = []
            tgt.extend([SOO])  
            for l in f['output']:
                tgt.extend(l + [NR,])
            tgt.extend([EOO])
            tgts.append(tgt)
            # if len(test) >1:import pdb;pdb.set_trace()
    return srcs, tgts

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
ts = []
tt = []

# add original trainset
empty_args = copy.deepcopy(args)
empty_args.shuffle_train=False
empty_args.shuffle_row=False
empty_args.random_rotate = False
empty_args.random_hue = False
empty_args.random_truncate = False
empty_args.times=1
empty_args.train_only=False

origs, origt = gen_seq('training', empty_args)
ts += origs
tt += origt

# augment with trainset
full_args = copy.deepcopy(args)
full_args.train_only = False

for i in range(args.times):
    # break
    s, t = gen_seq('training', full_args)
    ts += s
    tt += t
    # import pdb;pdb.set_trace()

# augment with train of all set
to_args = copy.deepcopy(args)
to_args.train_only = True
# folders = ['training', 'evaluation', 'test']
folders = ['training', 'evaluation']
# folders = ['training']
# folders = []
for folder in folders:
    for i in range(args.times):
        s, t = gen_seq(folder, to_args)
        ts += s
        tt += t

# valid set
vs, vt = gen_seq('evaluation', empty_args)

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
