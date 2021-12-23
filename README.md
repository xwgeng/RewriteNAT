# RewriteNAT
This repo provides the code for reproducing our proposed RewriteNAT in EMNLP 2021 paper entitled "[Learning to Rewrite for Non-Autoregressive Neural Machine Translation](https://aclanthology.org/2021.emnlp-main.265)". RewriteNAT is a iterative NAT model which utilizes a locator component to explicitly learn to rewrite the erroneous translation pieces during iterative decoding.
<p align="center">
  <img src="architecture.png">
</p>

## Dependencies
* [Pytorch](https://github.com/pytorch/pytorch) = 1.2
* [Fairseq](https://github.com/pytorch/fairseq) = 0.9

## Preprocessing
All the datasets are tokenized using the scripts from [Moses](https://github.com/moses-smt/mosesdecoder) except for Chinese with [Jieba tokenizer](https://github.com/fxsjy/jieba), and splitted into subword units using [BPE](https://github.com/rsennrich/subword-nmt). The tokenized datasets are binaried using the script `binaried.sh` as follows:
```bash
python preprocess.py \
    --source-lang ${src} --target-lang ${tgt} \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/${dataset} --thresholdtgt 0 --thresholdsrc 0 \ 
    --workers 64 --joined-dictionary
```

## Train
All the models are run on 8 Tesla V100 GPUs for 300,000 updates with an effective batch size of 128,000 tokens apart from En→Fr where we make 500,000 updates to account for the data size. The training scripts `train.rewrite.nat.sh` is configured as follows:
```bash
python train.py \
    data-bin/${dataset} \
    --source-lang ${src} --target-lang ${tgt} \
    --save-dir ${save_dir} \
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
```

## Evaluation
We evaluate performance with [BLEU](https://aclanthology.org/P02-1040) for all language pairs, except for En->Zh, where we use [SacreBLEU](https://www.aclweb.org/anthology/W18-6319). The testing scripts `test.rewrite.nat.sh` is utilized to generate the translations, as follows:
```bash
python generate.py \                                            
    data-bin/${dataset} \                                          
    --source-lang ${src} --target-lang ${tgt} \                    
    --gen-subset ${subset} \                                       
    --task translation_lev \                                       
    --path ${save_dir}/${dataset}/checkpoint_average_${suffix}.pt \
    --iter-decode-max-iter ${max_iter} \                           
    --iter-decode-with-beam ${beam} \                              
    --iter-decode-p ${iter_p} \                                    
    --beam 1 --remove-bpe \                                        
    --batch-size 50\                                               
    --print-step \                                                 
    --quiet 
```

## Citation
Please cite as:

```bibtex
@inproceedings{geng-etal-2021-learning,
    title = "Learning to Rewrite for Non-Autoregressive Neural Machine Translation",
    author = "Geng, Xinwei and Feng, Xiaocheng and Qin, Bing",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.265",
    pages = "3297--3308",
}
```
