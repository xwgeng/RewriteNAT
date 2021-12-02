# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq.models import register_model, register_model_architecture
from fairseq.models.nat.nonautoregressive_transformer import NATransformerModel, NATransformerDecoder
from fairseq.models.transformer import Embedding, TransformerDecoderLayer
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import new_arange
from fairseq.modules import GradMultiply

from fairseq.models.nat import ensemble_decoder


def _skeptical_unmasking(output_scores, output_masks, p):
    sorted_index = output_scores.sort(-1)[1]
    boundary_len = (
        (output_masks.sum(1, keepdim=True).type_as(output_scores) - 2) * p
    ).long()
    skeptical_mask = new_arange(output_masks) < boundary_len
    return skeptical_mask.scatter(1, sorted_index, skeptical_mask)


@register_model("rewrite_nonautoregressive_transformer")
class RewriteNATransformerModel(NATransformerModel):
    @staticmethod
    def add_args(parser):
        NATransformerModel.add_args(parser)

        # --- L2R special arguments ---
        parser.add_argument("--no-share-discriminator", action="store_true",
                            help="seperate parameters for discriminator")
        parser.add_argument("--discriminator-layers", type=int, metavar="N",
                            help="num discriminator layers")
        parser.add_argument("--train-max-iter", type=int, metavar="N",
                            help="maximum of refinement iterations during training")
        parser.add_argument("--adaptive-iter", action="store_true",
                            help="adaptive refinement iterations during training")
        parser.add_argument("--imitation-g", action="store_true",
                            help="apply imitation learning into generator")
        parser.add_argument("--imitation-d", action="store_true",
                            help="apply imitation learning into discriminator")
        parser.add_argument("--roll-in-g", choices=["sample", "max"],
                            help="roll-in policy to train generator")
        parser.add_argument("--roll-in-d", choices=["sample", "max", "mask", "oracle"],
                            help="roll-in policy to train discriminator ")
        parser.add_argument('--discriminator-loss-factor', type=float, metavar='D',
                            help='weights on discriminator loss')
        parser.add_argument('--generator-scale', type=float, metavar='D',
                            help='scale for gradients of generator module')
        parser.add_argument('--discriminator-scale', type=float, metavar='D',
                            help='scale for gradients of discriminator module')
        parser.add_argument("--generate-masking", action="store_true",
                            help="apply masking mechanism into generator module")
        parser.add_argument("--discriminate-oracle-masking", action="store_true",
                            help="apply oracle masking mechanism into discriminator masking")
        parser.add_argument("--discriminate-oracle-sampling", action="store_true",
                            help="apply oracle masking mechanism into discriminator sampling")
        parser.add_argument("--discriminate-masking", action="store_true",
                            help="apply masking mechanism into discriminator module")
        parser.add_argument("--discriminate-gap", type=float, metavar='D',
                            help="prob gap for applying masking mechanism into discriminator module")
        parser.add_argument("--discriminate-first-step", action="store_true",
                            help="calculate discriminator loss only at first iteration")
        parser.add_argument("--discriminator-gamma", type=float, metavar='D',
                            help="gamma for scaling discriminator loss at different iterations")

    @classmethod
    def build_model(cls, args, task):
        model = super().build_model(args, task)
        model.train_max_iter = args.train_max_iter
        model.adaptive_iter = args.adaptive_iter
        model.imitation_g = args.imitation_g
        model.imitation_d = args.imitation_d
        model.roll_in_g = args.roll_in_g
        model.roll_in_d = args.roll_in_d
        model.generator_scale = getattr(args, "generator_scale", 1.0)
        model.discriminator_scale = getattr(args, "discriminator_scale", 1.0)
        model.generate_masking = getattr(args, "generate_masking", False)
        model.discriminate_oracle_masking = getattr(args, "discriminate_oracle_masking", False)
        model.discriminate_oracle_sampling = getattr(args, "discriminate_oracle_sampling", False)
        model.discriminate_masking = getattr(args, "discriminate_masking", False)
        model.discriminate_gap = getattr(args, "discriminate_gap", 1.0)
        model.discriminate_first_step = getattr(args, "discriminate_first_step", False)
        model.discriminator_gamma = getattr(args, "discriminator_gamma", 1.0)
        return model

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        decoder = L2RNATransformerDecoder(args, tgt_dict, embed_tokens)
        if getattr(args, "apply_bert", False):
            decoder.apply(init_bert_params)
        return decoder

    def forward(
        self, src_tokens, src_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):
        assert not self.decoder.src_embedding_copy, "do not support embedding copy."

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, **kwargs)
        # length prediction
        length_out = self.decoder.forward_length(normalize=False, encoder_out=encoder_out)
        length_tgt = self.decoder.forward_length_prediction(length_out, encoder_out, tgt_tokens)

        # decoding
        generator_outs, generator_tgts, generator_masks = [], [], []
        discriminator_outs, discriminator_tgts, discriminator_masks = [], [], []
        discriminator_accs = []

        output_tokens = prev_output_tokens
        #output_scores = output_tokens.new_zeros(*output_tokens.size()).type_as(encoder_out.encoder_out)

        discriminator_gamma_list = [pow(self.discriminator_gamma, k) for k in range(self.train_max_iter)]
        discriminator_gamma_sum = sum(discriminator_gamma_list)
        discriminator_gamma_list = [v / discriminator_gamma_sum for v in discriminator_gamma_list]
        for t in range(self.train_max_iter):
            generator_out = self.decoder.forward_generator(
                normalize=False,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out
            )
            generator_tgt = tgt_tokens
            generator_mask = output_tokens.eq(self.unk)

            if self.discriminator_gamma != 1.0:
                generator_out = GradMultiply.apply(generator_out, discriminator_gamma_list[t])
            
            generator_outs.append(generator_out)
            generator_tgts.append(generator_tgt)
            generator_masks.append(generator_mask)

            generator_score = F.softmax(generator_out, -1)
            if self.roll_in_g == 'sample':
                generator_sample = torch.multinomial(
                    generator_score.view(-1, generator_score.size(-1)), 1
                ).view(generator_score.size(0), -1)
            elif self.roll_in_g == 'max':
                generator_sample = generator_score.max(-1)[1]
            else:
                raise ValueError('Can not use {} as roll-in policy for generator'.format(self.roll_in_g))

            output_tokens = output_tokens.masked_scatter(generator_mask, generator_sample[generator_mask])
            output_tokens.detach_()
            #output_scores.masked_scatter_(generator_mask, generator_score[generator_mask])

            discriminator_out = self.decoder.forward_discriminator(
                normalize=False,
                prev_output_tokens=output_tokens,
                encoder_out=encoder_out
            )
            discriminator_tgt = output_tokens.ne(tgt_tokens).type_as(tgt_tokens)
            discriminator_mask = output_tokens.ne(self.pad)

            discriminator_score = F.softmax(discriminator_out, -1)
            discriminator_max = discriminator_score.max(dim=-1)[1]
            discriminator_acc = discriminator_max.eq(discriminator_tgt) & discriminator_mask
            discriminator_acc = discriminator_acc.type_as(discriminator_score).sum() / discriminator_mask.type_as(discriminator_score).sum()
            discriminator_accs.append(discriminator_acc)

            if self.discriminate_oracle_masking:
                discriminator_mask = discriminator_mask & (output_tokens.tgt_tokens)

            if self.discriminate_masking:
                discriminator_val = discriminator_score.gather(
                    -1, discriminator_tgt.unsqueeze(-1)).squeeze(-1)
                discriminator_gap = discriminator_val - (1 - discriminator_val)
                discriminator_mask = discriminator_mask & discriminator_gap.lt(self.discriminate_gap)

            if self.generate_masking:
                discriminator_mask = generator_mask & discriminator_mask    
            
            if not self.discriminate_first_step or (self.discriminate_first_step and t == 0):
                discriminator_outs.append(discriminator_out)
                discriminator_tgts.append(discriminator_tgt)
                discriminator_masks.append(discriminator_mask)

            if self.roll_in_d == 'sample':
                discriminator_sample = torch.multinomial(
                    discriminator_score.view(-1, discriminator_score.size(-1)), 1
                ).view(discriminator_score.size(0), -1).eq(1)
            elif self.roll_in_d == 'max':
                discriminator_sample = discriminator_score.max(-1)[1].eq(1)
            #elif self.roll_in_d == 'mask':
            #    discriminator_prediction = _skeptical_unmasking(
            #        output_scores, 
            #        output_tokens.ne(self.pad), 
            #        1 - 1.0 * (t + 1) / self.train_max_iter
            #    )
            elif self.roll_in_d == 'oracle':
                discriminator_sample = output_tokens.ne(tgt_tokens)
            else:
                raise ValueError('Can not use {} as roll-in policy for discriminator'.format(self.roll_in_d))
            discriminator_sample = discriminator_mask & discriminator_sample # discriminator_tgt.type_as(discriminator_mask)

            if self.discriminate_oracle_sampling:
                discriminator_sample = output_tokens.ne(tgt_tokens) & discriminator_sample

            output_tokens = output_tokens.masked_fill(discriminator_sample, self.unk)
            output_tokens.detach_()
            #output_scores.masked_fill_(discriminator_prediction, 0.0)

            if self.adaptive_iter:
                discriminator_mask = output_tokens.ne(self.pad)
                discriminator_tgt = discriminator_tgt.type_as(discriminator_mask) & discriminator_mask

                not_same_sample = discriminator_sample.ne(generator_mask).sum(dim=-1).gt(0)
                not_halt_sample = discriminator_sample.sum(dim=-1).gt(0)
                not_same_tgt = discriminator_tgt.ne(generator_mask).sum(dim=-1).gt(0)
                not_halt_tgt = discriminator_tgt.sum(dim=-1).gt(0)
                not_terminated = not_same_sample & not_halt_sample & not_same_tgt & not_halt_tgt

                discriminator_outs[-1]= discriminator_outs[-1][not_same_tgt]
                discriminator_tgts[-1] = discriminator_tgts[-1][not_same_tgt]
                discriminator_masks[-1] = discriminator_masks[-1][not_same_tgt]

                if any(not_terminated) == False:
                    break

                output_tokens = output_tokens[not_terminated]
                #output_scores = output_scores[non_terminated]
                tgt_tokens = tgt_tokens[not_terminated]
                encoder_out = self.encoder.reorder_encoder_out(encoder_out, not_terminated.nonzero().squeeze())

        output = {
            "length": {
                "out": length_out, "tgt": length_tgt,
                "factor": self.decoder.length_loss_factor
            }
        }
        if self.imitation_g:
            generator_out = torch.cat(generator_outs, 0)
            generator_tgt = torch.cat(generator_tgts, 0)
            generator_mask = torch.cat(generator_masks, 0)

            if self.generator_scale != 1.0:
                generator_out = GradMultiply.apply(generator_out, self.generator_scale)
            
            output["generator"] = {
                "out": generator_out, "tgt": generator_tgt,
                "mask": generator_mask, "ls": self.args.label_smoothing,
                "nll_loss": True
            }

        if self.imitation_d:
            discriminator_out = torch.cat(discriminator_outs, 0)
            discriminator_tgt = torch.cat(discriminator_tgts, 0)
            discriminator_mask = torch.cat(discriminator_masks, 0)

            if self.discriminator_scale != 1.0:
                discriminator_out = GradMultiply.apply(discriminator_out, self.discriminator_scale)

            output["discriminator"] = {
                "out": discriminator_out, "tgt": discriminator_tgt,
                "mask": discriminator_mask, "factor": self.decoder.discriminator_loss_factor,
                "acc": discriminator_accs
            }

        return output
            
    def forward_decoder(self, decoder_out, encoder_out, decoding_format=None, **kwargs):
        step = decoder_out.step
        max_step = decoder_out.max_step
        iter_p = decoder_out.iter_p

        output_tokens = decoder_out.output_tokens
        output_scores = decoder_out.output_scores
        history = decoder_out.history

        # execute the decoder
        generator_mask = output_tokens.eq(self.unk)
        generator_score, generator_pred = self.decoder.forward_generator(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out,
        ).max(-1)
        output_tokens.masked_scatter_(generator_mask, generator_pred[generator_mask])
        output_scores.masked_scatter_(generator_mask, generator_score[generator_mask])

        if history is not None:
            history.append(output_tokens.clone())

        if (step + 1) >= max_step:
            return decoder_out._replace(
                output_tokens=output_tokens,
                output_scores=output_scores,
                history=history,
            )

        #discriminator_pred = _skeptical_unmasking(
        #    output_scores, output_tokens.ne(self.pad), 1 - (step + 1) / max_step
        #)
        discriminator_pred = self.decoder.forward_discriminator(
            normalize=True,
            prev_output_tokens=output_tokens,
            encoder_out=encoder_out
        )
        discriminator_pred = discriminator_pred[:,:,1].gt(iter_p)
        discriminator_mask = output_tokens.ne(self.pad)
        discriminator_pred = discriminator_pred & discriminator_mask

        if history is not None:
            history.append(output_tokens.masked_fill(discriminator_pred, self.unk))

        not_same_pred = discriminator_pred.ne(generator_mask).sum(dim=-1).gt(0)
        not_halt_pred = discriminator_pred.sum(dim=-1).gt(0)
        not_terminated = not_same_pred & not_halt_pred

        if any(not_terminated) == True:
            discriminator_pred = discriminator_pred & not_terminated.unsqueeze(-1)
            output_tokens.masked_fill_(discriminator_pred, self.unk)
            output_scores.masked_fill_(discriminator_pred, 0.0)
        
        return decoder_out._replace(
            output_tokens=output_tokens,
            output_scores=output_scores,
            history=history,
            terminated=~not_terminated
        )

    def load_state_dict(self, state_dict, strict=True, args=None):
        new_state_dict = {}
        keys = state_dict.keys()
        contain_keys = False
        for k in keys:
            if ('layers' in k) and ('decoder' in k):
                new_k = k.replace("layers", "discriminator")
                if new_k not in keys:
                    new_state_dict[new_k] = state_dict[k].clone().detach_()
                else:
                    contain_keys = True
                    break
            new_state_dict[k] = state_dict[k]
        if not contain_keys:
            output = super().load_state_dict(new_state_dict, strict=False, args=args)
        else:
            output = super().load_state_dict(state_dict, strict=True, args=args)
        print(output)
        #for n, p in self.named_parameters():
        #    if n.startswith('encoder'):
        #        p.requires_grad = False
        return output


class RewriteNATransformerDecoder(NATransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(
            args, dictionary, embed_tokens, no_encoder_attn=no_encoder_attn
        )
        self.discriminator_layers = args.discriminator_layers
        self.no_share_discriminator = args.no_share_discriminator
        self.embed_discriminator = Embedding(2, self.output_embed_dim, None)
        self.discriminator_loss_factor = args.discriminator_loss_factor

        self.discriminator = None
        if self.no_share_discriminator:
            self.discriminator = nn.ModuleList([
                TransformerDecoderLayer(args, no_encoder_attn)
                for _ in range(self.discriminator_layers)
            ])

    def extract_features(
        self, prev_output_tokens, encoder_out=None, early_exit=None, layers=None, **unused
    ):
        x, decoder_padding_mask = self.forward_embedding(prev_output_tokens)

        x = x.transpose(0, 1)
        attn = None
        inner_states = [x]
        
        layers = self.layers if layers is None else layers
        early_exit = len(layers) if early_exit is None else early_exit
        for i, layer in enumerate(layers[:early_exit]):
            x, attn = layer(
                x,
                encoder_out.encoder_out if encoder_out is not None else None,
                encoder_out.encoder_padding_mask if encoder_out is not None else None,
                self_attn_mask=None,
                self_attn_padding_mask=decoder_padding_mask
            )
            inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        x = x.transpose(0, 1)
        
        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": attn, "inner_states": inner_states}

    @ensemble_decoder
    def forward_generator_discriminator(self, normalize, encoder_out, prev_output_tokens, **unused):
        assert not self.no_share_disciminator
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=len(self.layers),
            layers=self.layers,
            **unused
        )
        generator_out = self.output_layer(features)
        discriminator_out = F.linear(features, self.embed_discriminator.weight)
        if normalize:
            return F.log_softmax(generator_out, -1), F.log_softmax(discriminator_out, -1)
        else:
            return generator_out, discriminator_out

    @ensemble_decoder
    def forward_generator(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=len(self.layers),
            layers =self.layers,
            **unused
        )
        generator_out= self.output_layer(features)
        return F.log_softmax(generator_out, -1) if normalize else generator_out

    @ensemble_decoder
    def forward_discriminator(self, normalize, encoder_out, prev_output_tokens, **unused):
        features, _ = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            early_exit=self.discriminator_layers,
            layers=self.discriminator,
            **unused
        )
        discriminator_out = F.linear(features, self.embed_discriminator.weight)
        return F.softmax(discriminator_out, -1) if normalize else discriminator_out


@register_model_architecture("rewrite_nonautoregressive_transformer", "rewrite_nonautoregressive_transformer")
def rewrite_nonautoregressive_transformer_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    # --- special arguments ---
    args.sg_length_pred = getattr(args, "sg_length_pred", False)
    args.pred_length_offset = getattr(args, "pred_length_offset", False)
    args.length_loss_factor = getattr(args, "length_loss_factor", 0.1)
    args.ngram_predictor = getattr(args, "ngram_predictor", 1)
    args.src_embedding_copy = getattr(args, "src_embedding_copy", False)

    # --- L2R special arguments ---
    args.discriminator_layers = getattr(args, "discriminator_layers", args.decoder_layers)
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.train_max_iter = getattr(args, "train_max_iter", 10)
    args.adaptive_iter = getattr(args, "adaptive_iter", False)
    args.imitation_g = getattr(args, "imitation_g", False)
    args.imitation_d = getattr(args, "imitation_d", False)
    args.roll_in_g = getattr(args, "roll_in_g")
    args.roll_in_d = getattr(args, "roll_in_d")
    args.discriminator_loss_factor = getattr(args, "discriminator_loss_factor", 1.0)
    args.generator_scale = getattr(args, "generator_scale", 1.0)
    args.discriminator_scale = getattr(args, "discriminator_scale", 1.0)
    args.generate_masking = getattr(args, "generate_masking", False)
    args.discriminate_oracle_masking = getattr(args, "discriminate_oracle_masking", False)
    args.discriminate_oracle_sampling = getattr(args, "discriminate_oracle_sampling", False)
    args.discriminate_masking = getattr(args, "discriminate_masking", False)
    args.discriminate_gap = getattr(args, "discriminate_gap", 1.0)
    args.discriminate_first_step = getattr(args, "discriminate_first_step", False)
    args.discriminator_gamma = getattr(args, "discriminator_gamma", 1.0)


@register_model_architecture("rewrite_nonautoregressive_transformer", "rewrite_nonautoregressive_transformer_wmt_en_de")
def rewrite_nonautoregressive_transformer_wmt_en_de(args):
    rewrite_nonautoregressive_transformer_base_architecture(args)
