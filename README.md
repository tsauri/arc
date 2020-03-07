# arc

# pretrain

cd arc
TEXT=arcs1
python cprepare_arc.py  --times 30 --name ../datasets/$TEXT   --hue  --rotate --shuffle-train --flip --truncate
cd ../fairseq
fairseq-preprocess --source-lang src --target-lang tgt --trainpref ../datasets/$TEXT/train --validpref ../datasets/$TEXT/valid --testpref ../datasets/$TEXT/valid --destdir data-bin/$TEXT
NAME=a1
fairseq-train data-bin/$TEXT/  --max-source-positions 9000 --max-target-positions 9000 --arch arclc-e --save-dir chkpts/$NAME --batch-size 16  --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 5000 --no-epoch-checkpoints

# fine-tune
TEXT=arcf1
python vprepare_arc.py  --times 30 --name ../datasets/$TEXT   --hue  --rotate --shuffle-train --flip --truncate
cd ../fairseq
fairseq-preprocess --source-lang src --target-lang tgt --trainpref ../datasets/$TEXT/train --validpref ../datasets/$TEXT/valid --testpref ../datasets/$TEXT/valid --destdir data-bin/$TEXT
NAME=a1
cp -r chkpts/$NAME fchkpts/$NAME
fairseq-train data-bin/$TEXT/  --max-source-positions 9000 --max-target-positions 9000 --arch arclc-e --save-dir fchkpts/$NAME --batch-size 16  --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 5000 --no-epoch-checkpoints

# generate
fairseq-generate data-bin/$TEXT/     --path fchkpts/$NAME/checkpoint_last.pt     --batch-size 1 --beam 20 --gen-subset valid