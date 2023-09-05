#!/bin/bash

# USAGE: bash proc_colmap.sh <dir of images>

# python run_colmap.py $1 ${@:2}
# python create_split.py -y $1

OPERATION_ID="$(date '+%Y%m%d%H%M%S-%N')"
SESSION_NAME=VID_20230602_085920_00_011_office5

echo Launching experiment ${OPERATION_ID}

CKPT_DIR=../ckpt/${OPERATION_ID}
mkdir -p $CKPT_DIR
NOHUP_FILE=$CKPT_DIR/log
echo CKPT $CKPT_DIR
echo LOGFILE $NOHUP_FILE

CUDA_VISIBLE_DEVICES=0 python -u opt_colmap.py ../data/${SESSION_NAME}/colmap \
-t ${CKPT_DIR} -c configs/custom.json --dataset_type colmap
echo DETACH

echo RUN_RENDER
python render_imgs_circle.py ../ckpt/${OPERATION_ID}/ckpt.npz ../data/${SESSION_NAME}/colmap --dataset_type colmap \
--num_views 100 --traj_type circle --offset 0,0,0 --elevation 0 --radius 0.5
