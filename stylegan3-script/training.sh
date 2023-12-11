#!/bin/bash

# Clone repository
git clone https://github.com/NVlabs/stylegan3.git

# Setup module
module load python/gpu/3.10.5

# Install libraries
pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3

# Allocate resources on bigred200
srun -p gpu -A r00596 --gpus-per-node 2 --pty bash

# Update your lib path for attaching the login node python to bigred allocation
export PATH=$PATH:/N/u/<iu-user>/BigRed200/.local/bin

# Update CPLUS path for pyconfig.py comiplation as the module 3.10 does not have python2 files
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python2.7/"

# train model
## You can add --resume=training-runs/00009-stylegan3-t-snow-sanity-512x512-gpus2-batch32-gamma8.2/network-snapshot-000064.pkl
## for resuming the training
## --snap decide when your model will be stored
## dataset needs to be zipped img files of square dimnesions or a dataset tool can be used to get it in shape.
python stylegan3/train.py --outdir=training-runs --cfg=stylegan3-t --data=<path-to-dataset> --gpus=2 --batch=32 --gamma=8.2 --mirror=1 --workers=2 --snap=16 
