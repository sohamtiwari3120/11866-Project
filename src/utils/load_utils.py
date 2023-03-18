import cv2
import numpy as np
import os
import scipy
import pickle
from typing import Dict
from torchvision import transforms
import torch
from torch.autograd import Variable

EPSILON = 1e-10


def bilateral_filter(outputs):
    """ smoothing function

    function that applies bilateral filtering along temporal dim of sequence.
    """
    outputs_smooth = np.zeros(outputs.shape)
    for b in range(outputs.shape[0]):
        for f in range(outputs.shape[2]):
            smoothed = np.reshape(cv2.bilateralFilter(
                                  outputs[b,:,f], 5, 20, 20), (-1))
            outputs_smooth[b,:,f] = smoothed
    return outputs_smooth.astype(np.float32)


def create_data_vq(l_vq_model, speakerData_np, listenerData_np, audioData_np, transcriptData_np,
                   seq_len, startpoint=0, midpoint=None, data_type='on_logit',
                   btc=None, patch_size=8):
    """ data preparation function

    processes the data by truncating full input sequences to remove future info,
    and converts listener raw motion to listener codebook indices
    """

    speakerData = Variable(torch.from_numpy(speakerData_np),
                           requires_grad=False).cuda()
    listenerData = Variable(torch.from_numpy(listenerData_np),
                            requires_grad=False).cuda()
    audioData = Variable(torch.from_numpy(audioData_np),
                         requires_grad=False).cuda()
    transcriptData = Variable(torch.from_numpy(transcriptData_np),
                         requires_grad=False).cuda()

    ## future timesteps for speaker inputs (keep past and current context)
    speaker_full = speakerData[:,:(seq_len+patch_size),:]
    audio_full = audioData[:,:(seq_len+patch_size)*4,:]

    ## convert listener past inputs to codebook indices
    with torch.no_grad():
        if listenerData.dim() == 3:
            # if listener input is in the raw format, directly convert to indxs
            listener_past, listener_past_index = \
                        l_vq_model.module.get_quant(listenerData[:,:seq_len,:])
            btc = listener_past.shape[0], \
                  listener_past.shape[2], \
                  listener_past.shape[1]
            listener_past_index = torch.reshape(listener_past_index,
                                                (listener_past.shape[0], -1))
        else:
            # if listener input is already in index format, fetch the quantized
            # raw listener and then re-encode into a new set of indxs
            tmp_past_index = listenerData[:,:btc[1]]
            tmp_decoded = l_vq_model.module.decode_to_img(tmp_past_index, btc)
            new_past, new_past_index = l_vq_model.module.get_quant(
                                                    tmp_decoded[:,:seq_len,:])
            listener_past_index = torch.reshape(new_past_index,
                                                (new_past.shape[0], -1))

        ## dealing with future listener motion (during training only)
        listener_future = None
        listener_future_index = None
        if listenerData.shape[1] > seq_len:
            listener_future, listener_future_index = \
                        l_vq_model.module.get_quant(listenerData[:,seq_len:,:])
            listener_future_index = torch.reshape(listener_future_index,
                                                (listener_future.shape[0], -1))

    ## build input dictionary, which will be the input to the Predictor
    raw_listener = listenerData[:,seq_len:,:] if listenerData.dim() == 3 \
                    else None
    inputs = {"speaker_full": speaker_full,
              "listener_past": listener_past_index,
              "audio_full": audio_full,
              "transcript_full": transcriptData}
    return inputs, listener_future_index, raw_listener, btc

def load_transcripts(transcripts_dir_fp):
    """ Function to load all the transcript files and return a dictionary of full texts.

    Parameters
    ----------
    transcripts_dir_fp: str
        path to directory containing only transcript text files
    """
    transcript_fname_text_dict = {}
    for transcript_fp in os.listdir(transcripts_dir_fp):
        with open(os.path.join(transcripts_dir_fp, transcript_fp), "r") as f:
            transcript_fname_text_dict[os.path.basename(transcript_fp)] = {"full_text": f.read().strip()}
    print(f"Loaded {len(transcript_fname_text_dict)} transcripts from {transcripts_dir_fp}.")
    return transcript_fname_text_dict

def load_single_transcript_embedding(transcript_embedding_fp:str)->np.ndarray:
    """Util function to load a transcript embedding as a np array

    Args:
        transcript_embedding_fp (str): file path to the transcript embedidng npy file

    Returns:
        np.ndarray: np array containing embedding
    """    
    return np.load(transcript_embedding_fp).reshape(1, -1)

def load_all_transcript_embeddings(transcript_embeddings_dir: str)->Dict[str, torch.Tensor]:
    """Function to load all the transcript embeddings as torch tensors and return in dictionary where the keys are the transcript filenames and the values are the respective torch embeddings.

    Args:
        transcript_embeddings_dir (str): Directory containing all the transcript embeddings

    Returns:
        Dict[str, torch.Tensor]: Dictionary where the keys are the transcript filenames and the values are the respective torch embeddings.
    """
    if not os.path.exists(transcript_embeddings_dir):
        raise Exception(f"No transcript embeddings exist at {transcript_embeddings_dir}. Create the transcript embeddings using the desired transformer first. Refer utils/process_transcripts.py")
    filenames = os.listdir(transcript_embeddings_dir)
    transcript_embeddings_dict = {}
    for filename in filenames:
        transcript_embeddings_dict[filename[:-8]] = load_single_transcript_embedding(os.path.join(transcript_embeddings_dir, filename))
    return transcript_embeddings_dict

def load_test_data(config, pipeline, tag, out_num=0, vqconfigs=None,
                   smooth=False, speaker=None, segment_tag='', num_out=None):
    """ function to load test data from files

    Parameters
    ----------
    pipeline : str
        defines the type of data to be loaded 'er', (e: expression, r: rotation)
    tag: str
        specifies the file with the tag suffix to load from
    out_num: str
        specifies which postion the listener is in the video (left:0, right:1)
        used for definining prefix in file name
    vqconfigs: dict
        specifies the vqconfigs corresponding to the pretrained VQ-VAE
        used to load the std/mean info for listeners
    smooth: bool
        whether to use bilateral filtering to smooth loaded files
    speaker: str
        specifies the speaker name for whom we want to load data
    segment_tag: str
        another one of these prefix tags (not really used for public release)
    num_out: int
        used to specify how many segments to load (for debugging)
    """

    ## load all speaker information from files
    base_dir = config['data']['basedir']
    all_speakers = ['conan', 'fallon', 'kimmel', 'stephen', 'trevor'] \
                    if speaker is None else [speaker]
    test_X = None
    transcripts_embeddings_dict = load_all_transcript_embeddings(config['data']['transcript_embeddings_dir'])



    for speaker in all_speakers:
        fp = '{}/data/{}/test/p{}_speak_files_clean_deca{}.npy'\
                            .format(base_dir, speaker, 1-out_num, segment_tag)
        p0_fp = '{}/data/{}/test/p{}_speak_faces_clean_deca{}.npy'\
                            .format(base_dir, speaker, 1-out_num, segment_tag)
        p1_fp = '{}/data/{}/test/p{}_list_faces_clean_deca{}.npy'\
                            .format(base_dir, speaker, out_num, segment_tag)
        audio_fp = '{}/data/{}/test/p{}_speak_audio_clean_deca{}.npy'\
                            .format(base_dir, speaker, 1-out_num, segment_tag)
        tmp_filepaths = np.load(fp)
        p0_deca = np.load(p0_fp)
        tmp_X = p0_deca.astype(np.float32)[:,:,:56]
        tmp_Y = np.load(p1_fp).astype(np.float32)[:,:,:56]
        tmp_audio = np.load(audio_fp).astype(np.float32)
        if test_X is None:
            filepaths = tmp_filepaths
            test_X = tmp_X
            test_Y = tmp_Y
            test_audio = tmp_audio
        else:
            filepaths = np.concatenate((filepaths, tmp_filepaths), axis=0)
            test_X = np.concatenate((test_X, tmp_X), axis=0)
            test_Y = np.concatenate((test_Y, tmp_Y), axis=0)
            test_audio = np.concatenate((test_audio, tmp_audio), axis=0)
        print('loaded:', fp, filepaths.shape, tmp_filepaths.shape)
        print('*******')

    ## optional post processing steps on data
    if num_out is not None:
        filepaths = filepaths[:num_out,:,:]
        test_X = test_X[:num_out,:,:]
        test_Y = test_Y[:num_out,:,:]
        test_audio = test_audio[:num_out,:,:]
    if smooth:
        test_X = bilateral_filter(test_X)
        test_Y = bilateral_filter(test_Y)

    ## standardize dataset
    preprocess = np.load(os.path.join(config['model_path'],
                         '{}{}_preprocess_core.npz'.format(tag, pipeline)))
    body_mean_X = preprocess['body_mean_X']
    body_std_X = preprocess['body_std_X']
    body_mean_audio = preprocess['body_mean_audio']
    body_std_audio = preprocess['body_std_audio']
    # take the std/mean from the listener vqgan training
    y_preprocess = np.load(os.path.join('vqgan/',
        vqconfigs['l_vqconfig']['model_path'],'{}{}_preprocess_core.npz'\
            .format(vqconfigs['l_vqconfig']['tag'], pipeline)))
    body_mean_Y = y_preprocess['body_mean_Y']
    body_std_Y = y_preprocess['body_std_Y']
    std_info = {'body_mean_X': body_mean_X,
                'body_std_X': body_std_X,
                'body_mean_Y': body_mean_Y,
                'body_std_Y': body_std_Y}
    test_X = (test_X - body_mean_X) / body_std_X
    test_Y = (test_Y - body_mean_Y) / body_std_Y
    test_audio = (test_audio - body_mean_audio) / body_std_audio
    test_transcript_embs = []
    for filepath in filepaths[:][:, 0, 0]:
        test_transcript_embs.append(transcripts_embeddings_dict[os.path.basename(filepath)])
    test_transcript_embs = np.concatenate(test_transcript_embs, axis=0)
    return test_X, test_Y, test_audio, test_transcript_embs, filepaths, std_info


def load_data(config, pipeline, tag, rng, vqconfigs=None, segment_tag='',
              smooth=False):
    """ function to load train data from files

    see load_test_data() for associated parameters
    """

    base_dir = config['data']['basedir']
    out_num = 0
    transcripts_embeddings_dict = load_all_transcript_embeddings(config['data']['transcript_embeddings_dir'])

    if config['data']['speaker'] == 'all':
        ## load associated files for all speakers
        all_speakers = ['conan', 'kimmel', 'fallon', 'stephen', 'trevor']
        curr_paths = gt_windows = quant_windows = audio_windows = None
        for speaker in all_speakers:
            tmp_paths, tmp_gt, tmp_quant, tmp_audio, _ = \
                        get_local_files(base_dir, speaker, out_num, segment_tag)
            if curr_paths is None:
                curr_paths = tmp_paths
                gt_windows = tmp_gt
                quant_windows = tmp_quant
                audio_windows = tmp_audio
            else:
                curr_paths = np.concatenate((curr_paths, tmp_paths), axis=0)
                gt_windows = np.concatenate((gt_windows, tmp_gt), axis=0)
                quant_windows = np.concatenate((quant_windows, tmp_quant),
                                               axis=0)
                audio_windows = np.concatenate((audio_windows, tmp_audio),
                                               axis=0)
            print('path:', speaker)
            print('curr:',
                tmp_paths.shape, tmp_gt.shape, tmp_quant.shape, tmp_audio.shape)
    else:
        ## load specific files for specified speaker
        out_num = 1 if config['data']['speaker'] == 'fallon' else 0
        curr_paths, gt_windows, quant_windows, audio_windows, _ = \
            get_local_files(base_dir, config['data']['speaker'],
                            out_num, segment_tag)
    print('===> in/out',
            gt_windows.shape, quant_windows.shape, audio_windows.shape)

    ## Pre-processing of loaded data
    if smooth:
        gt_windows = bilateral_filter(gt_windows)
        quant_windows = bilateral_filter(quant_windows)
    # randomize train/test splits
    N = gt_windows.shape[0]
    train_N = int(N * 0.7)
    idx = np.random.permutation(N)
    train_idx, test_idx = idx[:train_N], idx[train_N:]
    train_X, test_X = gt_windows[train_idx, :, :].astype(np.float32),\
                      gt_windows[test_idx, :, :].astype(np.float32)
    train_Y, test_Y = quant_windows[train_idx, :, :].astype(np.float32),\
                      quant_windows[test_idx, :, :].astype(np.float32)
    train_audio, test_audio = audio_windows[train_idx, :, :].astype(np.float32),\
                              audio_windows[test_idx, :, :].astype(np.float32)
    print("====> train/test", train_X.shape, test_X.shape)

    ## check to see how to load/calculate std/dev
    body_mean_X, body_std_X, body_mean_Y, body_std_Y, \
        body_mean_audio, body_std_audio = calc_stats(config, vqconfigs, tag,
                                                     pipeline, train_X, train_Y,
                                                     train_audio)
    train_X = (train_X - body_mean_X) / body_std_X
    test_X = (test_X - body_mean_X) / body_std_X
    train_Y = (train_Y - body_mean_Y) / body_std_Y
    test_Y = (test_Y - body_mean_Y) / body_std_Y
    train_audio = (train_audio - body_mean_audio) / body_std_audio
    test_audio = (test_audio - body_mean_audio) / body_std_audio
    train_transcript_embs, test_transcript_embs = [], []
    for filepath in curr_paths[train_idx][:, 0, 0]:
        train_transcript_embs.append(transcripts_embeddings_dict[os.path.basename(filepath)])
    train_transcript_embs = np.concatenate(train_transcript_embs, axis=0)

    for filepath in curr_paths[test_idx][:, 0, 0]:
        test_transcript_embs.append(transcripts_embeddings_dict[os.path.basename(filepath)])
    test_transcript_embs = np.concatenate(test_transcript_embs, axis=0)

    
    print("=====> standardization done")
    return train_X, test_X, train_Y, test_Y, train_audio, test_audio, train_transcript_embs, test_transcript_embs


def get_local_files(base_dir, speaker, out_num, segment_tag):
    """ helper function for loading associated files """

    fp = '{}/data/{}/train/p{}_speak_files_clean_deca{}.npy'\
                .format(base_dir, speaker, 1-out_num, segment_tag)
    p0_fp = '{}/data/{}/train/p{}_speak_faces_clean_deca{}.npy'\
                .format(base_dir, speaker, 1-out_num, segment_tag)
    p1_fp = '{}/data/{}/train/p{}_list_faces_clean_deca{}.npy'\
                .format(base_dir, speaker, out_num, segment_tag)
    audio_fp = '{}/data/{}/train/p{}_speak_audio_clean_deca{}.npy'\
                .format(base_dir, speaker, 1-out_num, segment_tag)
    curr_paths = np.load(fp)
    p0_deca = np.load(p0_fp)
    gt_windows = p0_deca[:,:,:56]
    quant_windows = np.load(p1_fp)[:,:,:56]
    audio_windows = np.load(audio_fp)
    app_windows = p0_deca[:,:,56:]
    print('loaded...', speaker)
    return curr_paths, gt_windows, quant_windows, audio_windows, app_windows


def calc_stats(config, vqconfigs, tag, pipeline, train_X, train_Y, train_audio):
    """ helper function to calculate std/mean for different cases """
    if vqconfigs is not None:
        # if vqconfig is defined, use std/mean from VQ-VAE for listener
        y_preprocess = np.load(os.path.join('vqgan/',
            vqconfigs['l_vqconfig']['model_path'],'{}{}_preprocess_core.npz'\
                            .format(vqconfigs['l_vqconfig']['tag'], pipeline)))
        body_mean_Y = y_preprocess['body_mean_Y']
        body_std_Y = y_preprocess['body_std_Y']
        # then calculate std/mean for speaker motion + audio
        body_mean_X, body_std_X = mean_std_swap(train_X)
        body_mean_audio, body_std_audio = mean_std_swap(train_audio)
        np.savez_compressed(config['model_path'] + \
                            '{}{}_preprocess_core.npz'.format(tag, pipeline),
            body_mean_X=body_mean_X, body_std_X=body_std_X,
            body_mean_audio=body_mean_audio, body_std_audio=body_std_audio)
    else:
        # if vqconfig not defined, no prior mean/std info exists
        body_mean_X, body_std_X = mean_std_swap(train_X)
        body_mean_Y, body_std_Y = mean_std_swap(train_Y)
        body_mean_audio, body_std_audio = mean_std_swap(train_audio)
        assert body_mean_X.shape[0] == 1 and body_mean_X.shape[1] == 1
        np.savez_compressed(config['model_path'] + \
                            '{}{}_preprocess_core.npz'.format(tag, pipeline),
            body_mean_X=body_mean_X, body_std_X=body_std_X,
            body_mean_Y=body_mean_Y, body_std_Y=body_std_Y,
            body_mean_audio=body_mean_audio, body_std_audio=body_std_audio)
    return body_mean_X, body_std_X, body_mean_Y, body_std_Y, \
           body_mean_audio, body_std_audio

def mean_std_swap(data):
    """ helper function to calc std and mean """
    B,T,F = data.shape
    mean = data.mean(axis=1).mean(axis=0)[np.newaxis,np.newaxis,:]
    std =  data.std(axis=1).std(axis=0)[np.newaxis,np.newaxis,:]
    std += EPSILON
    return mean, std
