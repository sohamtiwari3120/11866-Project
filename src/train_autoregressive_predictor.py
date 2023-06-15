import argparse
import json
import logging
import numpy as np
import os
import scipy.io as sio

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision

from modules.fact_model import setup_model, calc_logit_loss
from vqgan.vqmodules.gan_models import setup_vq_transformer
from utils.base_model_util import *
from utils.load_utils import *


def gather_data(config, X, Y, audio, transcript_embs, l_vq_model, patch_size, seq_len, bi):
    """ method to prepare data into proper format for training

    Parameters
    ----------
    X: tensor (B,T1,F)
        Past+current raw speaker motion of sequence length T1
    Y: tensor (B,T2,F)
        Past raw listener motion of sequence length T2
    audio: tensor (B,T3,A)
        Past raw speaker audio of sequence length T3
    transcript_embs: tensor (B, TE)
        Embeddings of the associated text transcription, of dimension TE (Text Embeddings)
    l_vq_model:
        pre-trained VQ-VAE model used to discretize the past listener motion and
        decode future listener motion predictions
    patch_size: int
        patch length that we divide seq_len into for the VQ-VAE model
    seq_len: int
        full length of sequence that is taken as input into the VQ-VAE model
    bi: int
        current batch index
    """
    idxStart = bi * config['batch_size']
    speakerData_np = X[idxStart:(idxStart + config['batch_size']), :, :]
    listenerData_np = Y[idxStart:(idxStart + config['batch_size']), :, :]
    audioData_np = audio[idxStart:(idxStart + config['batch_size']), :, :]
    transcriptData_np = np.repeat(np.expand_dims(transcript_embs[idxStart:(idxStart + config['batch_size']), :], axis=1), config["fact_model"]["speaker_full_transformer_config"]["sequence_length"], axis=1)
    inputs, listener_future, raw_listener, btc = \
        create_data_vq(l_vq_model, speakerData_np, listenerData_np,
                        audioData_np, transcriptData_np, seq_len,
                        data_type=config['loss_config']['loss_type'],
                        patch_size=patch_size)
    return inputs, listener_future, raw_listener, btc


def generator_train_step(config, epoch, autoregressive_generator, g_optimizer, l_vq_model,
                         train_X, train_Y, train_audio, train_transcript_embs, rng, writer,
                         patch_size, seq_len):
    """ method to prepare data into proper format for training

    see gather_data() for remaining parameter definitions

    Parameters
    ----------
    epoch: int
    autoregressive_generator:
        Predictor model that outputs future listener motion conditioned on past
        listener motion and speaker past+current audio+motion
    g_optimizer:
        optimizer for training the Predictor model
    """

    autoregressive_generator.train()
    batchinds = np.arange(train_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)
    avgLoss = 0

    for bii, bi in enumerate(batchinds):
        inputs, listener_future, _, _ = gather_data(config, train_X, train_Y,
                                                    train_audio, train_transcript_embs, l_vq_model,
                                                    patch_size, seq_len, bi)
        prediction = autoregressive_generator(inputs,
                        config['fact_model']['cross_modal_model']['max_mask_len'],
                        -1)
        cut_point = listener_future.shape[1]
        logit_loss = calc_logit_loss(prediction[:,:cut_point,:],
                                     listener_future[:,:cut_point])
        g_loss = logit_loss
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step_and_update_lr()
        avgLoss += g_loss.detach().item()
        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            avgLoss / totalSteps, np.exp(avgLoss / totalSteps)))
            avgLoss = 0
    writer.add_scalar('Loss/train_totalLoss', avgLoss / totalSteps, epoch)


def generator_val_step(config, epoch, autoregressive_generator, g_optimizer, l_vq_model,
                       test_X, test_Y, test_audio, test_transcript_embs, currBestLoss,
                       prev_save_epoch, tag, writer, patch_size, seq_len):
    """ method to validate training of Predictor model

    see generator_train_step() for full parameters definition
    """

    autoregressive_generator.eval()
    batchinds = np.arange(test_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    testLoss = 0

    for bii, bi in enumerate(batchinds):
        inputs, listener_future, _, _ = gather_data(config, test_X, test_Y,
                                                    test_audio, test_transcript_embs, l_vq_model,
                                                    patch_size, seq_len, bi)
        with torch.no_grad():
            prediction = autoregressive_generator(inputs,
                config['fact_model']['cross_modal_model']['max_mask_len'], -1)
        cut_point = listener_future.shape[1]
        logit_loss = calc_logit_loss(prediction[:,:cut_point,:],
                                     listener_future[:,:cut_point])
        g_loss = logit_loss
        testLoss += g_loss.detach().item()

    testLoss /= totalSteps
    print('val_Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            testLoss, np.exp(testLoss)))
    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', testLoss / totalSteps, epoch)

    ## save model if the curent loss is better than previous best
    if testLoss < currBestLoss:
        prev_save_epoch = epoch
        checkpoint = {'config': args.config,
                      'state_dict': autoregressive_generator.state_dict(),
                      'optimizer': {
                        'optimizer': g_optimizer._optimizer.state_dict(),
                        'n_steps': g_optimizer.n_steps,
                      },
                      'epoch': epoch}
        fileName = config['model_path'] + \
                    '{}{}_best.pth'.format(tag, config['pipeline'])
        currBestLoss = testLoss
        torch.save(checkpoint, fileName)
        print('>>>> saving best epoch {}'.format(epoch), testLoss)
    return currBestLoss, prev_save_epoch, testLoss


def main(args):
    """ full pipeline for training the Predictor model """
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    print('using config', args.config)
    with open(args.config) as f:
      config = json.load(f)
    tag = config['tag']
    pipeline = config['pipeline']
    writer = SummaryWriter('runs/debug_{}{}'.format(tag, pipeline))
    args.get_attn = False
    currBestLoss = 1e3
    prev_save_epoch = 0
    ## can modify via configs, these are default for released model
    patch_size = 8
    seq_len = 32

    ## setting up the listener VQ-VAE and Predictor models
    # load pre-trained VQ-VAE model
    with open(config['l_vqconfig']) as f:
        l_vqconfig = json.load(f)
    l_model_path = 'vqgan/' + l_vqconfig['model_path'] + \
            '{}{}_best.pth'.format(l_vqconfig['tag'], l_vqconfig['pipeline'])
    l_vq_model, _, _, style_transfer = setup_vq_transformer(args, l_vqconfig,
                                            load_path=l_model_path)

    # NOTE: freezing the layers of the VQ-VAE model, so that the codebook's discrete representations do not change during training.
    for param in l_vq_model.parameters():
        if not style_transfer:
            param.requires_grad = False
        else:
            if "style_transfer_layer" not in param.name:
                param.requires_grad = False
    l_vq_model.eval()
    vq_configs = {'l_vqconfig': l_vqconfig, 's_vqconfig': None}
    # set up Predictor model
    fileName = config['model_path'] + \
                    '{}{}_best.pth'.format(tag, config['pipeline'])
    load_path = fileName if os.path.exists(fileName) else None
    autoregressive_generator, g_optimizer, start_epoch = setup_model(config, l_vqconfig,
                                                      s_vqconfig=None,
                                                      load_path=load_path, use_text_transcriptions=args.use_text_transcriptions,
                                                      use_concat_attention = args.use_concat_attention,
                                                      disable_strict_load=args.disable_strict_load)
    autoregressive_generator.train()

    ## training process
    train_ratio = config['data']['train_split_ratio'] if args.train_split_ratio is None else args.train_split_ratio
    train_X, val_X, train_Y, val_Y, train_audio, val_audio, train_transcript_embs, val_transcript_embs= \
        load_data(config, pipeline, tag, rng, vqconfigs=vq_configs,
                  segment_tag=config['segment_tag'], smooth=True, train_ratio=train_ratio)

    patience = config["early_stopping"]["patience"]
    num_epochs_since_loss_improv = 0

    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print('epoch', epoch, 'num_epochs', config['num_epochs'])
        if (epoch == start_epoch+config['num_epochs']-1) or (num_epochs_since_loss_improv == patience):
            if (num_epochs_since_loss_improv == patience):
                print('early stopping at:', epoch)
            else:
                print('training finished. stopping at:', epoch)
            print("prev save epoch: ", prev_save_epoch)
            print('best loss:', currBestLoss)
            break
        generator_train_step(config, epoch, autoregressive_generator, g_optimizer, l_vq_model,
                             train_X, train_Y, train_audio, train_transcript_embs, rng, writer,
                             patch_size, seq_len)
        currBestLoss, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, autoregressive_generator, g_optimizer, l_vq_model,
                               val_X, val_Y, val_audio, val_transcript_embs, currBestLoss,
                               prev_save_epoch, tag, writer, patch_size, seq_len)
        if currBestLoss == g_loss:
            num_epochs_since_loss_improv = 0
        else:
            num_epochs_since_loss_improv += 1

    print('final best loss:', currBestLoss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--train_split_ratio', type=float, default=None)
    parser.add_argument('-ut', '--use_text_transcriptions', action='store_true')
    parser.add_argument('-uc', '--use_concat_attention', action='store_true')
    parser.add_argument('-dsl', '--disable_strict_load', action='store_true')
    args = parser.parse_args()
    main(args)
