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
    and converts (past) listener raw motion to listener codebook indices
    """

    speakerData = Variable(torch.from_numpy(speakerData_np),
                           requires_grad=False).cuda()
    listenerData = Variable(torch.from_numpy(listenerData_np),
                            requires_grad=False).cuda() # (batch_size, window_len, d_m+3)?
    audioData = Variable(torch.from_numpy(audioData_np),
                         requires_grad=False).cuda()
    transcriptData = Variable(torch.from_numpy(transcriptData_np),
                         requires_grad=False).cuda()

    ## future timesteps for speaker inputs (keep past and current context)
    speaker_full = speakerData[:,:(seq_len+patch_size),:]
    audio_full = audioData[:,:(seq_len+patch_size)*4,:]
    # NOTE: Above, in line with the results from the paper on "Feedback delays can enhance anticipatory synchronization in human machine interaction", small delay in the feedback received from the actions of the driver can help the driver learn and anticipate response from the environment.
    # Thus in this case, for the current speaker input, we receive listener feedback/output which is lagging behind by patch_size, 'w'

    # WARNING: DON'T FORGET TO UNDERSTAND BELOW
    # NOTE: patch_size = 'w' = 8
    # NOTE: sequence_len = 't' = 32 = tau * w
    # NOTE: tau = 't' / 'w' = 32 / 8 = 4
    # Therefore speaker input is usually of length 32 + 8 = t + w = tau*w + w = (tau + 1) * w
    # Listener input is usally lagging behind by length 'w', hence is of length 't' = tau * w


    ## convert listener past inputs to codebook indices
    with torch.no_grad():
        if listenerData.dim() == 3:
            # if listener input is in the raw format (batch_size, window_len, d_m+3), directly convert to indxs
            listener_past, listener_past_index = \
                        l_vq_model.module.get_quant(listenerData[:,:seq_len,:])
            # only quantizing listener input of length 't', because we need to have the listener input lag behind by the speaker by length 'w'
            # import pdb; pdb.set_trace()
            #  listener_past shape (batch_size, tau = t/w, dz)
            btc = listener_past.shape[0], \
                  listener_past.shape[2], \
                  listener_past.shape[1]
            listener_past_index = torch.reshape(listener_past_index,
                                                (listener_past.shape[0], -1)) # (batch_size, t/w, 1)?
            # print(listener_past.shape, listener_past_index.shape)
        else:
            # if listener input is already in index format (batch_size, t/w, 1)?, fetch the quantized
            # raw listener and then re-encode into a new set of indxs
            # btc = [batch_size, dz, t/w]
            tmp_past_index = listenerData[:,:btc[1]] # so essentially the same input shape as on line 65?
            tmp_decoded = l_vq_model.module.decode_to_img(tmp_past_index, btc) # (batch_size, t, dm+3)?
            new_past, new_past_index = l_vq_model.module.get_quant(
                                                    tmp_decoded[:,:seq_len,:])
            listener_past_index = torch.reshape(new_past_index,
                                                (new_past.shape[0], -1)) # (batch_size, t/w, 1)?

        ## dealing with future listener motion (during training only)
        listener_future = None
        listener_future_index = None
        if listenerData.shape[1] > seq_len:
            # ideally listenerData.shape[1] == seq_len + patch_size
            listener_future, listener_future_index = \
                        l_vq_model.module.get_quant(listenerData[:,seq_len:,:]) # (batch_size, t/w,  dz); (batch_size, t/w, 1)?
            listener_future_index = torch.reshape(listener_future_index,
                                                (listener_future.shape[0], -1))

    ## build input dictionary, which will be the input to the Autoregressive Predictor
    raw_listener = listenerData[:,seq_len:,:] if listenerData.dim() == 3 \
                    else None
    inputs = {"speaker_full": speaker_full, # current raw speaker facial embeddings
              "listener_past": listener_past_index, # index seqeuences of past listener motion, lagging behind by 1 patch_len, i.e., lagging behind the speaker embeddings by 'w'
              "audio_full": audio_full, # current raw speaker audio embeddings
              "transcript_full": transcriptData # the text transcript embeddings are duplicated seq_len + patch_len number of times, i.e., t + w = tau*w + w = (tau + 1) * w
              }
    # listener_future_index - (batch_size, win_len, 1) OR (batch_size, 1, 1) if patch wise encoded
    # raw_listener - (batch_size, tau*w, 56) raw listener facial embeddings, before passing through VQ-VAE encoder and quantization
    # btc - not sure

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
    print("Loading transcripts from " + os.path.abspath(transcript_embeddings_dir))
    filenames = os.listdir(transcript_embeddings_dir)
    # print(transcript_embeddings_dir, len(filenames))
    transcript_embeddings_dict = {}
    for filename in filenames:
        transcript_embeddings_dict[filename.replace(".txt.npy", "")] = load_single_transcript_embedding(os.path.join(transcript_embeddings_dir, filename))
    return transcript_embeddings_dict

def load_test_data(config, pipeline, tag, out_num=0, vqconfigs=None,
                   smooth=False, speaker=None, segment_tag='', num_out=None):
    """ function to load test data from test audio, facil and text embedding files

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
    test_transcripts_embeddings_dict = load_all_transcript_embeddings(config['data']['test_transcript_embeddings_dir'])
    transcripts_segmented = config['data']['transcripts_segmented']

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
    for filepath_array in filepaths[:]:
        filepath = filepath_array[0, 0]
        if transcripts_segmented:
            start_idx = filepath_array[0, -1]
            end_idx = filepath_array[-1, -1]
            test_transcript_embs.append(test_transcripts_embeddings_dict[f"p1_{os.path.basename(filepath)}_{start_idx}_{end_idx}"])
        else:
            test_transcript_embs.append(test_transcripts_embeddings_dict[os.path.basename(filepath)])
    test_transcript_embs = np.concatenate(test_transcript_embs, axis=0)
    return test_X, test_Y, test_audio, test_transcript_embs, filepaths, std_info

def load_reference_style_embeddings(config, type:str):
    if type not in ["less", "more"]:
        raise ValueError("type must be either 'less' or'more'")
    pkl_files_dir = config['data'][f'{type}_expressive_style_embeddings_dir']
    pkl_files = [os.path.join(pkl_files_dir, f) for f in os.listdir(pkl_files_dir) if f.endswith('.pkl')]
    pkl_data = []
    for file in pkl_files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            expr_data = data['exp'].cpu().detach().numpy().squeeze()
            pose_data = data['pose'].cpu().detach().numpy().squeeze()
            pkl_data.append(np.concatenate([expr_data, pose_data], axis=0))
    unstd_data = np.array(pkl_data)
    # mean, stddev = mean_std_swap(unstd_data)
    mean =  unstd_data.mean(axis=0)[None, :]
    stddev = unstd_data.std(axis=0)[None, :] + EPSILON
    std_data = (unstd_data - mean) / stddev
    return std_data, mean, stddev

def format_reference_style_embeddings(embeddings:np.ndarray, seq_len=64, batch_size=32):
    """function to format the reference style embeddings into a format that can be used by the model

    Args:
        embeddings (_type_): _description_
        seq_len (_type_, optional): _description_. Defaults to 64.
        batch_size (_type_, optional): _description_. Defaults to 32.

    Returns:
        _type_: _description_
    """
    batched_embeddings = []
    i = 0
    n_emb = len(embeddings)
    temp_array = []
    while True:
        if i == n_emb:
            i = 0
        temp_array.append(embeddings[i])
        i+=1
        if len(temp_array) == seq_len:
            batched_embeddings.append(temp_array)
            temp_array = []
        if len(batched_embeddings) == 1:
            break
    res = np.repeat(np.array(batched_embeddings), repeats=batch_size, axis=0)
    assert res.shape == (batch_size, seq_len, 56), f"Invalid np array constructed - {res.shape}"
    return res

def load_data(config, pipeline, tag, rng, vqconfigs=None, segment_tag='',
              smooth=False, train_ratio=1.0):
    """function to load train and val data splits from TRAIN audio, facial and text embeddings
    see load_test_data() for associated parameters

    Args:
        config (_type_): _description_
        pipeline (_type_): _description_
        tag (_type_): _description_
        rng (_type_): _description_
        vqconfigs (_type_, optional): _description_. Defaults to None.
        segment_tag (str, optional): _description_. Defaults to ''.
        smooth (bool, optional): _description_. Defaults to False.
        train_ratio (float, optional): eg. if train_ratio=0.7, then train_split = 70% of TRAIN DATA and val_split=30% of TRAIN DATA. Defaults to 1.0.

    Returns:
        _type_: _description_
    """


    base_dir = config['data']['basedir']
    out_num = 0

    # the directory contains the sentence embeddings of the form -
    # p1_<youTubeVideoName>_<start_frame>_<end_frame>.txt.npy (if segemented)
    # OR <youTubeVideoName>.txt.npy (if not segmented)
    train_transcripts_embeddings_dict = load_all_transcript_embeddings(config['data']['train_transcript_embeddings_dir'])

    # NOTE: We are only loading train transcript embddings dict because this function will load the train and val splits (depending on the train_ratio input arg).
    transcripts_segmented = config['data']['transcripts_segmented']

    if config['data']['speaker'] == 'all':
        ## load associated files for all speakers
        all_speakers = ['conan', 'kimmel', 'fallon', 'stephen', 'trevor']
        curr_paths =  None
        gt_windows =  None
        quant_windows =  None
        audio_windows = None
        for speaker in all_speakers:
            tmp_paths, tmp_gt, tmp_quant, tmp_audio, _ = \
                        get_local_files(base_dir, speaker, out_num, segment_tag, "train")
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
                            out_num, segment_tag, "train")
    print('===> in/out',
            gt_windows.shape, quant_windows.shape, audio_windows.shape)

    ## Pre-processing of loaded data
    if smooth:
        gt_windows = bilateral_filter(gt_windows)
        quant_windows = bilateral_filter(quant_windows)

    # randomize train/val splits
    N = gt_windows.shape[0]
    if train_ratio > 1 or train_ratio < 0:
        train_ratio = 0.7
    print(f"The train_ratio = {train_ratio}")
    train_N = int(N * train_ratio)
    idx = np.random.permutation(N)
    train_idx, val_idx = idx[:train_N], idx[train_N:]
    train_X, val_X = gt_windows[train_idx, :, :].astype(np.float32),\
                      gt_windows[val_idx, :, :].astype(np.float32)
    train_Y, val_Y = quant_windows[train_idx, :, :].astype(np.float32),\
                      quant_windows[val_idx, :, :].astype(np.float32)
    train_audio, val_audio = audio_windows[train_idx, :, :].astype(np.float32),\
                              audio_windows[val_idx, :, :].astype(np.float32)
    print("====> train/val", train_X.shape, val_X.shape)

    ## check to see how to load/calculate std/dev
    body_mean_X, body_std_X, body_mean_Y, body_std_Y, \
        body_mean_audio, body_std_audio = calc_stats(config, vqconfigs, tag,
                                                     pipeline, train_X, train_Y,
                                                     train_audio)
    train_X = (train_X - body_mean_X) / body_std_X
    val_X = (val_X - body_mean_X) / body_std_X
    train_Y = (train_Y - body_mean_Y) / body_std_Y
    val_Y = (val_Y - body_mean_Y) / body_std_Y
    train_audio = (train_audio - body_mean_audio) / body_std_audio
    val_audio = (val_audio - body_mean_audio) / body_std_audio
    train_transcript_embs = []
    val_transcript_embs = []
    remove_index = []
    for i, filepath_array in enumerate(curr_paths[train_idx]):
        # import pdb; pdb.set_trace()
        filepath = filepath_array[0, 0]
        # if filepath == "conan_videos/done_conan_videos9/009YouTube":
        #     remove_index.append(i)
        #     continue
        if transcripts_segmented:
            start_idx = filepath_array[0, -1]
            end_idx = filepath_array[-1, -1]
            key = f"p1_{os.path.basename(filepath)}_{start_idx}_{end_idx}"
            train_transcript_embs.append(train_transcripts_embeddings_dict[key])
        else:
            train_transcript_embs.append(train_transcripts_embeddings_dict[os.path.basename(filepath)])
    # if len(remove_index) > 0:
    #     print(f"Removing {remove_index}")
    #     print(f"Shape of train_X: {train_X.shape}")
    #     print(f"Shape of train_Y: {train_Y.shape}")
    #     print(f"Shape of train_audio: {train_audio.shape}")
    #     train_X = np.delete(train_X, remove_index, axis=0)
    #     train_Y = np.delete(train_Y, remove_index, axis=0)
    #     train_audio = np.delete(train_audio, remove_index, axis=0)
    #     print(f"After Removing {remove_index}")
    #     print(f"Shape of train_X: {train_X.shape}")
    #     print(f"Shape of train_Y: {train_Y.shape}")
    #     print(f"Shape of train_audio: {train_audio.shape}")


    train_transcript_embs = np.concatenate(train_transcript_embs, axis=0)

    remove_index = []
    for i, filepath_array in enumerate(curr_paths[val_idx]):
        filepath = filepath_array[0, 0]
        # if filepath == "conan_videos/done_conan_videos9/009YouTube":
        #     remove_index.append(i)
        #     continue
        if transcripts_segmented:
            start_idx = filepath_array[0, -1]
            end_idx = filepath_array[-1, -1]
            key = f"p1_{os.path.basename(filepath)}_{start_idx}_{end_idx}"
            val_transcript_embs.append(train_transcripts_embeddings_dict[key])
        else:
            val_transcript_embs.append(train_transcripts_embeddings_dict[os.path.basename(filepath)])
    val_transcript_embs = np.concatenate(val_transcript_embs, axis=0) if len(val_transcript_embs) > 0 else np.array(val_transcript_embs)
    # if len(remove_index) > 0:
    #     print(f"Removing {remove_index}")
    #     print(f"Shape of val_X: {val_X.shape}")
    #     print(f"Shape of val_Y: {val_Y.shape}")
    #     print(f"Shape of val_audio: {val_audio.shape}")
    #     val_X = np.delete(val_X, remove_index, axis=0)
    #     val_Y = np.delete(val_Y, remove_index, axis=0)
    #     val_audio = np.delete(val_audio, remove_index, axis=0)
    #     print(f"After Removing {remove_index}")
    #     print(f"Shape of val_X: {val_X.shape}")
    #     print(f"Shape of val_Y: {val_Y.shape}")
    #     print(f"Shape of val_audio: {val_audio.shape}")

    print("=====> standardization done")
    return train_X, val_X, train_Y, val_Y, train_audio, val_audio, train_transcript_embs, val_transcript_embs


def get_local_files(base_dir, speaker, out_num, segment_tag, split="train"):
    """ helper function for loading associated files """

    fp = '{}/data/{}/{}/p{}_speak_files_clean_deca{}.npy'\
                .format(base_dir, speaker, split, 1-out_num, segment_tag)
    p0_fp = '{}/data/{}/{}/p{}_speak_faces_clean_deca{}.npy'\
                .format(base_dir, speaker, split, 1-out_num, segment_tag)
    p1_fp = '{}/data/{}/{}/p{}_list_faces_clean_deca{}.npy'\
                .format(base_dir, speaker, split, out_num, segment_tag)
    audio_fp = '{}/data/{}/{}/p{}_speak_audio_clean_deca{}.npy'\
                .format(base_dir, speaker, split, 1-out_num, segment_tag)
    curr_paths = np.load(fp)
    p0_deca = np.load(p0_fp)
    # NOTE: WE ONLY ACCESS THE FIRST 56 VALUES BECAUSE THE FIRST 50 are EXPRESSIONS and the next 6 values are POSE INFORMATION
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
    # data (1, 169, 56)
    mean = data.mean(axis=1).mean(axis=0)[np.newaxis,np.newaxis,:] # (1, 1, 56)
    std =  data.std(axis=1).std(axis=0)[np.newaxis,np.newaxis,:]
    std += EPSILON
    return mean, std


if __name__ == '__main__':
    import json
    with open('/home/ubuntu/learning2listen/src/configs/vq/delta_v6.json', 'r') as f:
        config = json.load(f)
    less_style_embeddings, less_mean, less_stddev = load_reference_style_embeddings(config, "less")
    more_style_embeddings, more_mean, more_stddev = load_reference_style_embeddings(config, "more")
    less_style_embeddings_batch = format_reference_style_embeddings(less_style_embeddings)
    more_style_embeddings_batch = format_reference_style_embeddings(more_style_embeddings)
    import pdb; pdb.set_trace()