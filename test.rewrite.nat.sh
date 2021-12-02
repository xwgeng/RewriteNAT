#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

dataset=$2

data=${dataset%.*}

lang=${data#*.}
src=${lang%-*}
tgt=${lang#*-}

save_dir=$3

subset=$4

max_iter=$5

beam=$6
beam=${beam:-1}

suffix=$7

iter_p=$8
iter_p=${iter_p:-0.5}

PYTHON=/userhome/anaconda3/envs/pytorch1.2-release-py3.6/bin/python

${PYTHON} generate.py \
    data-bin/${dataset} \
    --source-lang ${src} --target-lang ${tgt} \
    --gen-subset ${subset} \
    --task translation_lev \
    --path ${save_dir}/${dataset}/checkpoint_average_${suffix}.pt \
    --iter-decode-max-iter ${max_iter} \
    --iter-decode-with-beam ${beam} \
    --iter-decode-p ${iter_p} \
    --beam 1 --remove-bpe \
    --batch-size 25\
    --print-step \
    --quiet \
