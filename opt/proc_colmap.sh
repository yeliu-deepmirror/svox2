#!/bin/bash

# USAGE: bash proc_colmap.sh <dir of images>

# python run_colmap.py $1 ${@:2}


# python scripts/colmap2nsvf.py --sparse_dir ../data/20220726T162250+0800_xvnxa_xvnxa001_gzns_2jmw2/colmap/sparse/


# python create_split.py -y $1

OPERATION_ID="$(date '+%Y%m%d%H%M%S-%N')"
SESSION_NAME=20220726T162250+0800_xvnxa_xvnxa001_gzns_2jmw2

echo Launching experiment ${OPERATION_ID}

CKPT_DIR=../ckpt/${OPERATION_ID}
mkdir -p $CKPT_DIR
NOHUP_FILE=$CKPT_DIR/log
echo CKPT $CKPT_DIR
echo LOGFILE $NOHUP_FILE

CUDA_VISIBLE_DEVICES=0 python -u opt_colmap.py ../data/${SESSION_NAME}/colmap \
-t ${CKPT_DIR} -c configs/custom.json --dataset_type colmap
echo DETACH


# <data_dir> -c configs/custom.json
