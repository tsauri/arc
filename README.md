# arc
python vprepare_arc.py  --times 50 --name ../datasets/arcfv4    --hue   --shuffle-train --rotate --flip  --truncate --repeat-frames --stretch
python cprepare_arc.py  --times 30 --name ../datasets/arcs1   --hue  --rotate --shuffle-train --flip --truncate
cd -
fairseq-preprocess --source-lang src --target-lang tgt --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/valid --destdir data-bin/arcfv4
fairseq-train data-bin/arcfv4/  --max-source-positions 9000 --max-target-positions 9000 --arch arclc-c --save-dir fchkpts/$NAME --batch-size 16  --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 5000 
CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/arcfv3 --max-source-positions 9000 --max-target-positions 9000     --path fchkpts/$NAME/checkpoint_last.pt --batch-size 1 --beam 20  --nbest 3 --gen-subset valid --print-alignment 
