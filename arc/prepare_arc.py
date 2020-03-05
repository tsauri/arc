import json
import os
import copy
NR = 10
SOI = 11
EOI = 12
SOO = 13
EOO = 14
def read_json(path_to_jsonfiles):
    alldicts = []
    for file in os.listdir(path_to_jsonfiles):
        full_filename = "%s/%s" % (path_to_jsonfiles, file)
        with open(full_filename,'r') as fi:
            dict = json.load(fi)
            alldicts.append(dict)
    return alldicts

# import pdb;pdb.set_trace()
# with open("train.txt", "w") as text_file:
def gen_seq(dirname, filename):
    dicts = read_json(dirname)
    # mto = [d for d in train_dicts if len(d['test']) != 1]
    # import pdb; pdb.set_trace()
    srcs = []
    tgts = []
    for idx, d in enumerate(dicts):
        train = d['train']
        test = d['test']
        src = []
        for f in train:
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
            temp_src = []

            temp_src.extend([SOI])
            # if len(test) >1:import pdb;pdb.set_trace()
            for l in f['input']:
                temp_src.extend(l + [NR,])
            temp_src.extend([EOI])
            srcs.append(copy.deepcopy(src + temp_src)) # we need to change src if not end seq
            # if len(test) >1:import pdb;pdb.set_trace()
            if 'output' not in f.keys(): 
                break # for now only predict 1 timestep
            tgt = []
            tgt.extend([SOO])  
            for l in f['output']:
                tgt.extend(l + [NR,])
            tgt.extend([EOO])
            tgts.append(tgt)
            # if len(test) >1:import pdb;pdb.set_trace()

    with open(filename+".src", "w") as src_text:
        for src in srcs:
            src = [str(i) for i in src]
            src_text.write(' '.join(src))
            src_text.write("\n")
    with open(filename+".tgt", "w") as tgt_text:
        for tgt in tgts:
            tgt = [str(i) for i in tgt]
            tgt_text.write(' '.join(tgt))
            tgt_text.write("\n")

    return srcs, tgts
os.makedirs('arc_raw', exist_ok=True)
gen_seq('training','arc_raw/train')
gen_seq('evaluation','arc_raw/valid')
# gen_seq('test','arc_raw/test')
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