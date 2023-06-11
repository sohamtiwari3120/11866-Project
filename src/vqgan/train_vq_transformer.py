"""
This file contains code to train the VQ-VAE encoder, the quantizer and the corresponding decoder (quant_index -> image) for the listener. This takes a lot of time to train. This is essentially to obtain the 'codebook' discrete latent representations for a listener.
"""

import argparse
import json
import logging
import numpy as np
import os
import scipy.io as sio

import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.utils.tensorboard import SummaryWriter

from vqmodules.gan_models import setup_vq_transformer, calc_vq_loss
import sys
sys.path.append('../')
from utils.load_utils import *


def generator_train_step(config, epoch, generator, g_optimizer, train_X,
                         rng, writer, more_style_embeddings_batch, less_style_embeddings_batch, style_transfer):
    """ Function to do autoencoding training for VQ-VAE

    Parameters
    ----------
    generator:
        VQ-VAE model that takes as input continuous listener and learns to
        outputs discretized listeners
    g_optimizer:
        optimizer that trains the VQ-VAE
    train_X:
        continuous listener motion sequence (acts as the target)
    """

    if style_transfer:
        # train_X = np.repeat(train_X, repeats=2, axis=0) # repeating every element twice to train once with more and less style embeddings respectively
        # [face1 face2  face3  face4  face5  face6  face7  face8]
        orig_len = train_X.shape[0]
        train_X = np.concatenate([train_X, train_X], axis=0)
        gt_more = Variable(torch.from_numpy(more_style_embeddings_batch),
                          requires_grad=False).cuda()
        gt_less = Variable(torch.from_numpy(less_style_embeddings_batch),
                          requires_grad=False).cuda()
        beta_style_transfer = config['style_transfer']['loss']['beta_style_transfer']
        token_more = torch.ones(1, 1)
        token_less = torch.ones(1, 1) * 0
        print("line 49", len(train_X) == train_X.shape[0])
        style_type = config['style_transfer']['style_type']
        print(f'style_type = {style_type}')

    generator.train()
    batchinds = np.arange(len(train_X) // config['batch_size'])
    totalSteps = len(batchinds)
    rng.shuffle(batchinds)
    avgLoss = avgDLoss = 0


    for bii, bi in enumerate(batchinds):
        idxStart = bi * config['batch_size']
        gtData_np = train_X[idxStart:(idxStart + config['batch_size']), :, :]
        gtData = Variable(torch.from_numpy(gtData_np),
                        requires_grad=False).cuda()
        if style_transfer:
            if style_type=="only_more" or (style_type=="both_more_less" and idxStart >= orig_len):
                # use more expressive style embeddings
                prediction, quant_loss = generator(gtData, None, style_token=token_more)
                style_transfer_loss = calc_vq_loss(prediction, gt_more, quant_loss)
            elif style_type=="only_less" or (style_type=="both_more_less" and idxStart < orig_len):
                # use less expressive style embeddings
                prediction, quant_loss = generator(gtData, None, style_token=token_less)
                style_transfer_loss = calc_vq_loss(prediction, gt_less, quant_loss)
            # if bii % 2 == 0:
            #     prediction, quant_loss = generator(gtData, None, style_token=token_less)
            #     style_transfer_loss = calc_vq_loss(prediction, gt_less, quant_loss)
            # else:
            #     prediction, quant_loss = generator(gtData, None, style_token=token_more)
            #     style_transfer_loss = calc_vq_loss(prediction, gt_more, quant_loss)
            # prediction, quant_loss = generator(gtData, None, style_token=corresponding_style_tokens[idxStart:(idxStart + config['batch_size'])])
            # style_transfer_loss = calc_vq_loss(prediction, corresponding_gt_styles[idxStart:(idxStart + config['batch_size'])], quant_loss)
        else:
            prediction, quant_loss = generator(gtData, None)
        g_loss = calc_vq_loss(prediction, gtData, quant_loss)

        if style_transfer:
            cumulative_loss = beta_style_transfer * style_transfer_loss + (1 - beta_style_transfer) * g_loss
        else:
            cumulative_loss = g_loss

        g_optimizer.zero_grad()
        cumulative_loss.backward()
        g_optimizer.step_and_update_lr()
        avgLoss += cumulative_loss.detach().item()

        if bii % config['log_step'] == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                    .format(epoch, config['num_epochs'], bii, totalSteps,
                            avgLoss / totalSteps, np.exp(avgLoss / totalSteps)))
            avgLoss = 0

    writer.add_scalar('Loss/train_totalLoss', avgLoss / totalSteps, epoch)


def generator_val_step(config, epoch, generator, g_optimizer, test_X,
                       currBestLoss, prev_save_epoch, tag, writer, more_style_embeddings_batch, less_style_embeddings_batch, style_transfer):
    """ Function that validates training of VQ-VAE

    see generator_train_step() for parameter definitions
    """
    if style_transfer:
        orig_len = test_X.shape[0]
        test_X = np.concatenate([test_X, test_X], axis=0) # repeating every element twice to train once with more and less style embeddings respectively
        # test_X = np.repeat(test_X, repeats=2, axis=0) # repeating every element twice to train once with more and less style embeddings respectively
        gt_more = Variable(torch.from_numpy(more_style_embeddings_batch),
                          requires_grad=False).cuda()
        gt_less = Variable(torch.from_numpy(less_style_embeddings_batch),
                          requires_grad=False).cuda()
        beta_style_transfer = config['style_transfer']['loss']['beta_style_transfer']
        token_more = torch.ones(1, 1)
        token_less = torch.ones(1, 1) * 0
        style_type = config['style_transfer']['style_type']
        print(f'style_type = {style_type}')

    generator.eval()
    batchinds = np.arange(test_X.shape[0] // config['batch_size'])
    totalSteps = len(batchinds)
    testLoss = testDLoss = 0

    for bii, bi in enumerate(batchinds):
        idxStart = bi * config['batch_size']
        gtData_np = test_X[idxStart:(idxStart + config['batch_size']), :, :]
        gtData = Variable(torch.from_numpy(gtData_np),
                          requires_grad=False).cuda()

        with torch.no_grad():
            # prediction, quant_loss = generator(gtData, None)
            if style_transfer:
                # if bii % 2 == 0:
                if style_type=="only_more" or (style_type=="both_more_less" and idxStart >= orig_len):
                    prediction, quant_loss = generator(gtData, None, style_token=token_more)
                    style_transfer_loss = calc_vq_loss(prediction, gt_more, quant_loss)
                elif style_type=="only_less" or (style_type=="both_more_less" and idxStart < orig_len):
                    prediction, quant_loss = generator(gtData, None, style_token=token_less)
                    style_transfer_loss = calc_vq_loss(prediction, gt_less, quant_loss)
            else:
                prediction, quant_loss = generator(gtData, None)
        g_loss = calc_vq_loss(prediction, gtData, quant_loss)

        if style_transfer:
            cumulative_loss = beta_style_transfer * style_transfer_loss + (1 - beta_style_transfer) * g_loss
        else:
            cumulative_loss = g_loss
        # g_loss = calc_vq_loss(prediction, gtData, quant_loss)
        testLoss += cumulative_loss.detach().item()
    testLoss /= totalSteps
    print('val_Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Perplexity: {:5.4f}'\
                .format(epoch, config['num_epochs'], bii, totalSteps,
                        testLoss, np.exp(testLoss)))
    print('----------------------------------')
    writer.add_scalar('Loss/val_totalLoss', testLoss / totalSteps, epoch)

    ## save model if curr loss is lower than previous best loss
    if testLoss < currBestLoss:
        prev_save_epoch = epoch
        checkpoint = {'config': args.config,
                      'state_dict': generator.state_dict(),
                      'optimizer': {
                        'optimizer': g_optimizer._optimizer.state_dict(),
                        'n_steps': g_optimizer.n_steps,
                      },
                      'epoch': epoch}
        fileName = config['model_path'] + \
                        '{}{}_best.pth'.format(tag, config['pipeline'])
        print('>>>> saving best epoch {}'.format(epoch), testLoss)
        currBestLoss = testLoss
        torch.save(checkpoint, fileName)
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
    currBestLoss = 1e8
    ## can modify via configs, these are default for released model
    seq_len = 32
    prev_save_epoch = 0
    style_transfer = config['VQuantizer']['style_transfer']
    freeze_codebook = config['VQuantizer']['freeze_codebook']
    if style_transfer:
        print('using style transfer')
        if freeze_codebook:
            print('>'*3+' '+'freezing codebook')

    writer = SummaryWriter('runs/debug_{}{}'.format(tag, pipeline))

    ## setting up models
    fileName = config['model_path'] + \
                '{}{}_best.pth'.format(tag, config['pipeline'])
    if args.load_path is None:
        load_path = fileName if os.path.exists(fileName) else None
    else:
        load_path = args.load_path
    generator, g_optimizer, start_epoch, style_transfer = setup_vq_transformer(args, config,
                                            version=None, load_path=load_path)
    generator.train()

    train_split_ratio = config['data']['train_split_ratio']
    ## training/validation process
    _, _, train_listener, test_listener, _, _, _, _ = load_data(config, pipeline, tag, rng,
                              segment_tag=config['segment_tag'], smooth=True, train_ratio=train_split_ratio)
    train_X = np.concatenate((train_listener[:,:seq_len,:],
                              train_listener[:,seq_len:,:]), axis=0)
    test_X = np.concatenate((test_listener[:,:seq_len,:],
                             test_listener[:,seq_len:,:]), axis=0)
    batch_size = config['batch_size']

    # load reference style embeddings
    less_style_embeddings, less_mean, less_stddev = load_reference_style_embeddings(config, "less")
    more_style_embeddings, more_mean, more_stddev = load_reference_style_embeddings(config, "more")
    less_style_embeddings_batch = format_reference_style_embeddings(less_style_embeddings, seq_len=seq_len, batch_size=batch_size)
    more_style_embeddings_batch = format_reference_style_embeddings(more_style_embeddings, seq_len=seq_len, batch_size=batch_size)


    print('loaded listener...', train_X.shape, test_X.shape)
    print(f'Loaded reference style embeddings: {more_style_embeddings_batch.shape} {less_style_embeddings_batch.shape}')
    disc_factor = 0.0
    for epoch in range(start_epoch, start_epoch + config['num_epochs']):
        print('epoch', epoch, 'num_epochs', config['num_epochs'])
        if epoch == start_epoch+config['num_epochs']-1:
            print('best loss:', currBestLoss)
            print('early stopping at:', epoch)
            break
        generator_train_step(config, epoch, generator, g_optimizer, train_X,
                             rng, writer, more_style_embeddings_batch, less_style_embeddings_batch, style_transfer)
        currBestLoss, prev_save_epoch, g_loss = \
            generator_val_step(config, epoch, generator, g_optimizer, test_X,
                               currBestLoss, prev_save_epoch, tag, writer, more_style_embeddings_batch, less_style_embeddings_batch, style_transfer)
    print('final best loss:', currBestLoss)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    # parser.add_argument('--data_config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ar_load', action='store_true')
    parser.add_argument('--load_path', type=str, default=None)
    args = parser.parse_args()
    main(args)
