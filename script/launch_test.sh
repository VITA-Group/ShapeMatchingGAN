#!/bin/bash
python test.py \
--text_name ../data/rawtext/yaheiB/val/0801.png \
--scale -1 --scale_step 0.2 \
--structure_model ../save/fire-GS-iccv.ckpt \
--texture_model ../save/fire-GT-iccv.ckpt \
--result_dir ../output --name fire-0801 \
--gpu