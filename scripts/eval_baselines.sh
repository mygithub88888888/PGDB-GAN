#!/usr/bin/env bash
# ============================================================================
# PGDB-GAN - Per-baseline evaluation manifest (reproducibility record)
#
# This file consolidates the evaluation entry points of every baseline method
# used in the paper, pinned to the exact official repository commit and the
# official pretrained checkpoint listed in the README ("Reproducibility"
# section, release v1.0.0). No baseline was retrained or fine-tuned: each
# official checkpoint is evaluated with the official code at the recorded
# commit, under the paper's data splits, preprocessing, resolutions, and
# metric protocol.
#
# Usage: run each block inside a fresh clone of the official repository
# checked out at the recorded commit. Replace <low_input_dir> and
# <output_dir> with the paper's dataset/test directories.
# ============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# LLNet (official color version; Theano, Python 2.7)
# repo: kglore/llnet_color @ 1d45245ec2f6439ffd67848e05daa104412e3755
# weights: model object (.obj) downloaded via the link in the repo README
# ---------------------------------------------------------------------------
# python llnet_color.py test [IMAGE_FILENAME] [MODEL_FILENAME]

# ---------------------------------------------------------------------------
# LightenNet
# NOTE: no public official code or checkpoint exists. The reported values are
# cited from the original LightenNet paper (see the reproducibility table in
# the README), so no execution block is provided.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Retinex-Net
# repo: weichen582/RetinexNet @ fdc15ebc179209d17c77371a825df351a5be3ff5
# weights: under ./checkpoint (per repo README)
# ---------------------------------------------------------------------------
python main.py --use_gpu=1 --phase=test --test_dir=<low_input_dir> --save_dir=<output_dir>

# ---------------------------------------------------------------------------
# MBLLEN
# repo: Lvfeifan/MBLLEN @ 69f6dc7ac35e4e1e5d79e74d2738cca033f5d563
# weights: Syn_img_lowlight_withnoise.h5, LOL_img_lowlight.h5
# ---------------------------------------------------------------------------
cd main
python test.py -i <low_input_dir> -r <output_dir> -m <model_name>

# ---------------------------------------------------------------------------
# KinD
# repo: zhangyhuaee/KinD @ b7d7fcca6d70e1fcb588ad6935ec7750e96c7161
# weights: official checkpoints (Baidu/Google Drive links in repo README)
# ---------------------------------------------------------------------------
python evaluate.py
# LOL split:
python evaluate_LOLdataset.py

# ---------------------------------------------------------------------------
# KinD++
# repo: zhangyhuaee/KinD_plus @ 6e50ecdbf092420276bf4cf18f7343110b20e17f
# weights: official checkpoints (Drive/Baidu links in repo README)
# ---------------------------------------------------------------------------
python evaluate.py
# LOL split:
python evaluate_LOLdataset.py

# ---------------------------------------------------------------------------
# TBEFN
# repo: lukun199/TBEFN @ c9181c7a4fc05a7f0050847a858c97268511701c
# weights: ./ckpt (provided in repo)
# ---------------------------------------------------------------------------
python predict_TBEFN.py        # TF 1.13
# or
python predict_TBEFN_tf2.py    # TF 2.x

# ---------------------------------------------------------------------------
# DSLR
# repo: SeokjaeLIM/DSLR-release @ 861429482faf50ee3d6570948af8c48df1fc7f43
# weights: pretrained model via Drive link in repo README
# ---------------------------------------------------------------------------
python test.py t

# ---------------------------------------------------------------------------
# EnlightenGAN
# repo: VITA-Group/EnlightenGAN @ b0349848f0cd1e52317baa04e09ac32a2ae771d6
# weights: pretrained generator + VGG16 (Drive links in repo README)
# ---------------------------------------------------------------------------
python scripts/script.py --predict

# ---------------------------------------------------------------------------
# DRBN
# repo: flyywh/CVPR-2020-Semi-Low-Light @ 9f383decbd2717ab37bb9e4c133b3a0bf98ba638
# weights: official checkpoints (per repo README)
# ---------------------------------------------------------------------------
sh ./DRBL-stage1/src/test.sh

# ---------------------------------------------------------------------------
# ExCNet
# repo: csLinZhang/ExCNet @ 440c3d8572658d3eab3a570cf9e35bfe06478953
# weights: none (the official notebook ExCNet.ipynb is executed as-is; the
# repo contains no separate pretrained checkpoint)
# ---------------------------------------------------------------------------
# Run the official notebook ExCNet.ipynb as-is.

# ---------------------------------------------------------------------------
# Zero-DCE
# repo: Li-Chongyi/Zero-DCE @ e0f4adc54d0f23348c4a9b84acc08fe8778d5bfd
# weights: Epoch99.pth
# ---------------------------------------------------------------------------
python lowlight_test.py

# ---------------------------------------------------------------------------
# RRDNet
# repo: aaaaangel/RRDNet @ d1dce2a2069777a64bd335c210cee91e0e03a86e
# weights: none required (zero-shot Retinex decomposition)
# ---------------------------------------------------------------------------
python3 pipline.py

# ---------------------------------------------------------------------------
# SCI
# repo: vis-opt-group/SCI @ f6f88fd73cd614dbeee17d61a0dbde3678b7e183
# weights: CVPR/weights/easy.pt, medium.pt, difficult.pt; TPAMI/weights/weights_1_3500.pt
# ---------------------------------------------------------------------------
python CVPR/test.py --data_path <low_input_dir> --save_path <output_dir> --model CVPR/weights/medium.pt --gpu 0
# (easy/medium/difficult per the difficulty split; use TPAMI/weights/weights_1_3500.pt for the TPAMI setting)

# ---------------------------------------------------------------------------
# SNR-Net
# repo: JIA-Lab-research/SNR-Aware-Low-Light-Enhance @ 1113144c82adc8bcc4a9ec27749ed75f196a4e4d
# weights: official download links in repo README; set pretrain_model_G in the
# yml below to the downloaded checkpoint
# ---------------------------------------------------------------------------
python test.py -opt options/test/<dataset>.yml

# ---------------------------------------------------------------------------
# Retinexformer
# repo: caiyuanhao1998/Retinexformer @ 1e9a0efce4b306b6701b824768370ff26066c32a
# weights: official Google Drive/Baidu download links in repo README
# ---------------------------------------------------------------------------
python3 Enhancement/test_from_dataset.py --opt Options/RetinexFormer_<DATASET>.yml --weights pretrained_weights/<DATASET>.pth --dataset <DATASET>