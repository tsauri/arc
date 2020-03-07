NAME=a2
TEXT=arcf1
cp -r chkpts/$NAME fchkpts/$NAME
fairseq-train data-bin/$TEXT/  --max-source-positions 9000 --max-target-positions 9000 --arch arclc-e --save-dir fchkpts/$NAME --batch-size 16  --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt --warmup-updates 5000 --me 30	