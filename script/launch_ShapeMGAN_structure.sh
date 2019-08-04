#!/bin/bash
python trainStructureTransfer.py \
--style_name ../data/style/fire.png \
--batchsize 16 --Straining_num 2560 \
--step1_epochs 30 --step2_epochs 40 --step3_epochs 80 \
--scale_num 4 \
--Sanglejitter \
--save_path ../save --save_name fire \
--text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
--load_GB_name ../save/GB-iccv.ckpt \
--gpu