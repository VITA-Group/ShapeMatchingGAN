#!/bin/bash
python trainTextureTransfer.py \
--style_name ../data/style/fire.png \
--batchsize 4 --Ttraining_num 800 \
--texture_step1_epochs 40 \
--Tanglejitter --style_loss \
--save_path ../save --save_name fire \
--text_path ../data/rawtext/yaheiB/train --text_datasize 708 \
--load_GS_name ../save/fire-GS.ckpt \
--gpu