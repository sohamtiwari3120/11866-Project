import functools
import json
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("../")
from vqgan.vqmodules.quantizer import VectorQuantizer, StyleTransferVectorQuantizer
from modules.base_models import Transformer, PositionEmbedding,\
                                LinearEmbedding, AudioEmbedding
from utils.base_model_util import get_activation
from utils.optim import ScheduledOptim


def setup_vq_transformer(args, config, load_path=None, test=False, version=None):
    """ function that creates and sets up the VQ-VAE model for train/test """

    ## create VQ-VAE model and the optimizer for training
    style_transfer = config['VQuantizer']['style_transfer']
    generator = VQModelTransformer(config, style_transfer=style_transfer).cuda()
    learning_rate =  config['learning_rate']
    print('starting lr', learning_rate)
    g_optimizer = ScheduledOptim(
            torch.optim.AdamW(generator.parameters(),
                              betas=(0.9, 0.98), eps=1e-09),
                              learning_rate,
                              config['transformer_config']['hidden_size'],
                              config['warmup_steps'])
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    generator = nn.DataParallel(generator)

    ## load model from prev checkpoint to resume training
    start_epoch = 0
    if load_path is not None:
        loaded_state = torch.load(load_path,
                                  map_location=lambda storage, loc: storage)
        if style_transfer:
            generator.load_state_dict(loaded_state['state_dict'], strict=False) # made such changes so that able to load pretrained codebook weights without raising an error
            # generator.module.quantize.load_pretrained_codebook_weights(load_path, freeze_codebook=config['VQuantizer']['freeze_codebook'])
            print(f"Loaded pretrained codebook weights from {load_path} for style transfer")
        else:
            generator.load_state_dict(loaded_state['state_dict'], strict=True) # made
        # g_optimizer._optimizer.load_state_dict(
        #                         loaded_state['optimizer']['optimizer'], strict=False)
        # g_optimizer.set_n_steps(loaded_state['optimizer']['n_steps'])
        # start_epoch = loaded_state['epoch']
        if loaded_state['epoch'] > 500:
            print('>> changing lr to 4.5e-06')
            g_optimizer.set_init_lr(4.5e-06)
        print('loading checkpoint from...', load_path)
        del loaded_state
    else:
        print('starting from scratch...')

    if config['VQuantizer']['freeze_codebook']:
        if style_transfer:
            generator.module.freeze_all_except_style_transfer_layer()
        else:
            generator.module.quantize.embedding.weight.requires_grad = False

        for name, param in generator.module.named_parameters():
            if param.requires_grad:
                print(name, param.data)

    print(f"Froze codebook: {config['VQuantizer']['freeze_codebook']}")

    return generator, g_optimizer, start_epoch, style_transfer


def calc_vq_loss(pred, target, quant_loss, quant_loss_weight=1.0, alpha=1.0):
    """ function that computes the various components of the VQ loss """

    exp_loss = nn.L1Loss()(pred[:,:,:50], target[:,:,:50]) # expression loss
    rot_loss = nn.L1Loss()(pred[:,:,50:53], target[:,:,50:53])
    jaw_loss = alpha * nn.L1Loss()(pred[:,:,53:], target[:,:,53:])
    # if quant_loss_weight == 0.0:
    #     import pdb; pdb.set_trace()
    ## loss is VQ reconstruction + weighted pre-computed quantization loss
    return quant_loss.mean() * quant_loss_weight + \
            (exp_loss + rot_loss + jaw_loss)


class VQModelTransformer(nn.Module):
    """ Transformer model for listener VQ-VAE """

    def __init__(self, config, style_transfer=False):
        super().__init__()
        self.encoder = TransformerEncoder(config)
        self.decoder = TransformerDecoder(
                                config, config['transformer_config']['in_dim'])
        self.style_transfer = style_transfer
        if self.style_transfer:
            self.quantize = StyleTransferVectorQuantizer(config['VQuantizer']['n_embed'],
                                            config['VQuantizer']['zquant_dim'],
                                            beta=0.25)
        else:
            self.quantize = VectorQuantizer(config['VQuantizer']['n_embed'],
                                        config['VQuantizer']['zquant_dim'],
                                        beta=0.25)

    def encode(self, x, x_a=None, style_token=None):
        h = self.encoder(x) ## x --> z'
        if self.style_transfer:
            quant, emb_loss, info = self.quantize(h, style_token) ## finds nearest quantization
        else:
            quant, emb_loss, info = self.quantize(h) ## finds nearest quantization
        return quant, emb_loss, info

    def freeze_all_except_style_transfer_layer(self):
        if self.style_transfer:
            self.quantize.freeze_codebook()
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            print(f"Froze all except style transfer layer")
        else:
            print(f"Cannot freeze, style transfer is {self.style_transfer}")

    def decode(self, quant):
        dec = self.decoder(quant) ## z' --> x
        return dec

    def forward(self, x, x_a=None, style_token=None):
        if self.style_transfer:
            quant, emb_loss, _ = self.encode(x, style_token=style_token)
        else:
            quant, emb_loss, _ = self.encode(x)
        dec = self.decode(quant)
        return dec, emb_loss

    def sample_step(self, x, x_a=None, style_token=None):
        if self.style_transfer:
            quant_z, _, info = self.encode(x, x_a, style_token=style_token)
        else:
            quant_z, _, info = self.encode(x, x_a)
        x_sample_det = self.decode(quant_z)
        btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
        indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check

    def get_quant(self, x, x_a=None, style_token=None):
        quant_z, _, info = self.encode(x, x_a, style_token=style_token)
        indices = info[2]
        return quant_z, indices

    def get_distances(self, x, style_token=None):
        h = self.encoder(x) ## x --> z'
        if self.style_transfer:
            d = self.quantize.get_distance(h, style_token)
        else:
            d = self.quantize.get_distance(h)
        return d

    def get_quant_from_d(self, d, btc):
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        x = self.decode_to_img(min_encoding_indices, btc)
        return x

    @torch.no_grad()
    def decode_to_img(self, index, zshape, style_token=None):
        index = index.long()
        if self.style_transfer:
            quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None, style_token=style_token)
        else:
            quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape).permute(0,2,1)
        x = self.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:
            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)
        else:
            ix = logits
        ix = torch.reshape(ix, (-1,1))
        x = self.decode_to_img(ix, zshape)
        return x

    def get_logit(self, logits, sample=True, filter_value=-float('Inf'),
                  temperature=0.7, top_p=0.9, sample_idx=None):
        """ function that samples the distribution of logits. (used in test)

        if sample_idx is None, we perform nucleus sampling
        """

        if sample_idx is None:
            ## nucleus sampling
            sample_idx = 0
            for b in range(logits.shape[0]):
                ## only take first prediction
                curr_logits = logits[b,0,:] / temperature
                assert curr_logits.dim() == 1
                sorted_logits, sorted_indices = torch.sort(curr_logits,
                                                           descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, \
                                                dim=-1), dim=-1)
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token
                # above the threshold
                sorted_indices_to_remove[..., 1:] = \
                                    sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                curr_logits[indices_to_remove] = filter_value
                logits[b,0,:] = curr_logits

        logits = logits[:,[0],:]
        probs = F.softmax(logits, dim=-1)
        if sample:
            ## multinomial sampling
            shape = probs.shape
            probs = probs.reshape(shape[0]*shape[1],shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx+1)[:,[-1]]
            probs = probs.reshape(shape[0],shape[1],shape[2])
            ix = ix.reshape(shape[0],shape[1],-1)
        else:
            ## top 1; no sampling
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs


class TransformerEncoder(nn.Module):
  """ Encoder class for VQ-VAE with Transformer backbone """

  def __init__(self, config):
    super().__init__()
    self.config = config
    size=self.config['transformer_config']['in_dim']
    dim=self.config['transformer_config']['hidden_size']
    layers = [nn.Sequential(
                   nn.Conv1d(size,dim,5,stride=2,padding=2,
                             padding_mode='replicate'),
                   nn.LeakyReLU(0.2, True),
                   nn.BatchNorm1d(dim))]
    for _ in range(1, config['transformer_config']['quant_factor']):
        layers += [nn.Sequential(
                       nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                 padding_mode='replicate'),
                       nn.LeakyReLU(0.2, True),
                       nn.BatchNorm1d(dim),
                       nn.MaxPool1d(2)
                       )]
    self.squasher = nn.Sequential(*layers)
    self.encoder_transformer = Transformer(
        in_size=self.config['transformer_config']['hidden_size'],
        hidden_size=self.config['transformer_config']['hidden_size'],
        num_hidden_layers=\
                self.config['transformer_config']['num_hidden_layers'],
        num_attention_heads=\
                self.config['transformer_config']['num_attention_heads'],
        intermediate_size=\
                self.config['transformer_config']['intermediate_size'])
    self.encoder_pos_embedding = PositionEmbedding(
        self.config["transformer_config"]["quant_sequence_length"],
        self.config['transformer_config']['hidden_size'])
    self.encoder_linear_embedding = LinearEmbedding(
        self.config['transformer_config']['hidden_size'],
        self.config['transformer_config']['hidden_size'])

  def forward(self, inputs):
    ## downdample into path-wise length seq before passing into transformer
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1)
    encoder_features = self.encoder_linear_embedding(inputs)
    encoder_features = self.encoder_pos_embedding(encoder_features)
    encoder_features = self.encoder_transformer((encoder_features, dummy_mask))
    return encoder_features


class TransformerDecoder(nn.Module):
  """ Decoder class for VQ-VAE with Transformer backbone """

  def __init__(self, config, out_dim, is_audio=False):
    super().__init__()
    self.config = config
    size=self.config['transformer_config']['hidden_size']
    dim=self.config['transformer_config']['hidden_size']
    self.expander = nn.ModuleList()
    self.expander.append(nn.Sequential(
                   nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                      output_padding=1,
                                      padding_mode='replicate'),
                   nn.LeakyReLU(0.2, True),
                   nn.BatchNorm1d(dim)))
    num_layers = config['transformer_config']['quant_factor']+2 \
        if is_audio else config['transformer_config']['quant_factor']
    seq_len = config["transformer_config"]["sequence_length"]*4 \
        if is_audio else config["transformer_config"]["sequence_length"]
    for _ in range(1, num_layers):
        self.expander.append(nn.Sequential(
                             nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                       padding_mode='replicate'),
                             nn.LeakyReLU(0.2, True),
                             nn.BatchNorm1d(dim),
                             ))
    self.decoder_transformer = Transformer(
        in_size=self.config['transformer_config']['hidden_size'],
        hidden_size=self.config['transformer_config']['hidden_size'],
        num_hidden_layers=\
            self.config['transformer_config']['num_hidden_layers'],
        num_attention_heads=\
            self.config['transformer_config']['num_attention_heads'],
        intermediate_size=\
            self.config['transformer_config']['intermediate_size'])
    self.decoder_pos_embedding = PositionEmbedding(
        seq_len,
        self.config['transformer_config']['hidden_size'])
    self.decoder_linear_embedding = LinearEmbedding(
        self.config['transformer_config']['hidden_size'],
        self.config['transformer_config']['hidden_size'])
    self.cross_smooth_layer=\
        nn.Conv1d(config['transformer_config']['hidden_size'],
                  out_dim, 5, padding=2)

  def forward(self, inputs):
    dummy_mask = {'max_mask': None, 'mask_index': -1, 'mask': None}
    ## upsample into original length seq before passing into transformer
    for i, module in enumerate(self.expander):
        inputs = module(inputs.permute(0,2,1)).permute(0,2,1)
        if i > 0:
            inputs = inputs.repeat_interleave(2, dim=1)
    decoder_features = self.decoder_linear_embedding(inputs)
    decoder_features = self.decoder_pos_embedding(decoder_features)
    decoder_features = self.decoder_transformer((decoder_features, dummy_mask))
    pred_recon = self.cross_smooth_layer(
                                decoder_features.permute(0,2,1)).permute(0,2,1)
    return pred_recon
