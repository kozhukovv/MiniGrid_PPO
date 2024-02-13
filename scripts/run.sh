#!/bin/sh

pip install -r requirements.txt

upgradever=0
newver=0.051

ver=$upgradever
while [ 1 = "$(echo "$ver < $newver" | bc -l)" ]
do
    python3 ./src/cnn_gru.py --exp_name gpu --num_steps 128 --ent_coef $ver --capture_video
    ver=$(echo "$ver + 0.002" | bc -l)
done
