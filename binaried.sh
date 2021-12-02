#!/bin/bash

set -e

export LC_ALL=en_US.UTF-8

dataset=$1

data=${dataset%.*}

lang=${data#*.}
src=${lang%-*}
tgt=${lang#*-}

PYTHON=/userhome/anaconda3/envs/pytorch1.2-release-py3.6/bin/python

TEXT=examples/translation/${dataset}
#TEXT=examples/translation/${dataset%.*}.${tgt}-${src}
${PYTHON} preprocess.py \
    --source-lang ${src} --target-lang ${tgt} \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/${dataset} --thresholdtgt 0 --thresholdsrc 0 \
    --workers 64 --joined-dictionary
