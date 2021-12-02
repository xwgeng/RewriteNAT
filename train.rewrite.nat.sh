#!/bin/bash

export CUDA_VISIBLE_DEVICES=$1

dataset=$2
max_iter=$3
discriminator_weight=$4
generator_scale=$5
discriminator_scale=$6
step=$7

data=${dataset%.*}

lang=${data#*.}
src=${lang%-*}
tgt=${lang#*-}

step=${step:-300000}

share_all_embeddings="--share-all-embeddings"
if [ ${lang} == "zh-en" ] || [ ${lang} == "en-zh" ];then
    share_all_embeddings=""
fi

save_interval=1

PYTHON=/userhome/anaconda3/envs/pytorch1.2-release-py3.6/bin/python

${PYTHON} train.py \
    data-bin/${dataset} \
    --source-lang ${src} --target-lang ${tgt} \
    --save-dir rewrite_iter${max_iter}_no_share_init_fixed_reset_lr_gsample_doracle_discriminator_${discriminator_weight}_generator_scale_${generator_scale}_discriminator_scale_${discriminator_scale}_128k_${step%000}k/${dataset} \
    --ddp-backend=no_c10d \
    --task translation_lev \
    --criterion rewrite_nat_loss \
    --arch rewrite_nonautoregressive_transformer \
    --noise full_mask \
    ${share_all_embeddings} \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 4000 \
    --save-interval-updates 10000 \
    --max-update ${step} \
    --update-freq 4 \
    --fp16 \
    --save-interval ${save_interval} \
    --discriminator-layers 6 \
    --train-max-iter ${max_iter} \
    --roll-in-g sample \
    --roll-in-d oracle \
    --imitation-g \
    --imitation-d \
    --discriminator-loss-factor ${discriminator_weight} \
    --no-share-discriminator \
    --generator-scale ${generator_scale} \
    --discriminator-scale ${discriminator_scale} \
    --restore-file cmlm_big_128k_300k/${dataset}/checkpoint_cmlm_128k.pt \
    --reset-optimizer \
    --reset-meters \
    --reset-dataloader \
    --reset-lr-scheduler \
    > log.rewrite_iter${max_iter}_no_share_init_fixed_reset_lr_gsample_doracle_discriminator_${discriminator_weight}_generator_scale_${generator_scale}_discriminator_scale_${discriminator_scale}_128k.${dataset}
