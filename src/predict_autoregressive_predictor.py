"""For creating predictions for single input video
"""

from test_autoregressive_predictor import *
import pandas as pd
import numpy as np

def save_gen_pred(args, config, tag, pipeline, save_folder, unstd_pred, probs=None):
    """ Method to saves predictions and probs to corresponding files """
    ## unstandardize outputs
    B,T,_ = unstd_pred.shape
    preprocess = np.load(os.path.join('vqgan/', config['model_path'],
                                '{}{}_preprocess_core.npz'.format(config['tag'],
                                config['pipeline'])))
    body_mean_Y = preprocess['body_mean_Y']
    body_std_Y = preprocess['body_std_Y']
    test_Y = unstd_pred * body_std_Y + body_mean_Y

    ## save predictions into corresponding files
    ctr = 0
    for b in range(B):
        for t in range(T):
            save_base = os.path.join(save_folder)
            if not os.path.exists(save_base):
                os.makedirs(save_base)
            save_path = os.path.join(save_base,
                                '{:08d}.pkl'.format(int(ctr)))
            ctr += 1
            data = {'exp': torch.from_numpy(test_Y[b,t,:50]).cuda()[None,...],
                    'pose': torch.from_numpy(test_Y[b,t,50:]).cuda()[None,...]}
            if probs is not None:
                data['prob'] = probs[b,int(t/8),:]
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
    print('done save', test_Y.shape)

def generate_prediction(args):
    speaker_video_embeddings = np.load(args.input_video_path)
    speaker_audio_embeddings = np.load(args.input_audio_path)
    speaker_text_embeddings = np.load(args.input_text_path)
    listener_video_embeddings = np.load("/home/ubuntu/learning2listen/src/data/conan/test/p0_list_faces_clean_deca.npy")[:speaker_video_embeddings.shape[0], :, :56]

    import pdb; pdb.set_trace()
    rng = np.random.RandomState(23456)
    torch.manual_seed(23456)
    torch.cuda.manual_seed(23456)
    seq_len = 32
    patch_size = 8
    num_out = 1024

    with open(args.config) as f:
      config = json.load(f)
    pipeline = config['pipeline']
    tag = config['tag']

    ## setup VQ-VAE model
    with open(config['l_vqconfig']) as f:
        l_vqconfig = json.load(f)
    l_model_path = 'vqgan/' + l_vqconfig['model_path'] + \
        '{}{}_best.pth'.format(l_vqconfig['tag'], l_vqconfig['pipeline'])
    l_vq_model, _, _, style_transfer = setup_vq_transformer(args, l_vqconfig,
                                            load_path=l_model_path,
                                            test=True)
    assert style_transfer == False, "Style transfer needs to be disabled"
    l_vq_model.eval()
    vq_configs = {'l_vqconfig': l_vqconfig, 's_vqconfig': None}

    ## setup Predictor model
    load_path = args.checkpoint
    print('> checkpoint', load_path)
    autoregressive_generator, _, _ = setup_model(config, l_vqconfig,
                                  mask_index=0, test=True, s_vqconfig=None,
                                  load_path=load_path, use_text_transcriptions=args.use_text_transcriptions, disable_strict_load=args.disable_strict_load)
    autoregressive_generator.eval()

    ## load data
    # print(f"Results for paths provided in {args.input_file_path}")
    test_X = np.concatenate(speaker_video_embeddings, axis=0)[None, :, :]
    test_Y = np.concatenate(listener_video_embeddings, axis=0)[None, :, :]
    test_audio = np.concatenate(speaker_audio_embeddings, axis=0)[None, :, :]
    test_transcript_embs = np.array([speaker_text_embeddings for _ in range(test_X.shape[0])])
    batch_size = test_X.shape[0]
    import pdb; pdb.set_trace()


    ## run model and save/eval
    unstd_pred, probs, unstd_ub = run_model(args, config, l_vq_model, autoregressive_generator,
                                            test_X, test_Y, test_audio, test_transcript_embs, seq_len,
                                            patch_size, rng=rng)
    subset = (test_X.shape[0] // batch_size) * batch_size
    if test_Y is not None:
        overall_l2 = np.mean(
            np.linalg.norm(test_Y[:subset,seq_len:,:] - unstd_pred[:,seq_len:,:], axis=-1))
        print(f"Listener - conan; description - {args.etag}\noverall l2: {overall_l2}\n{'-'*5}")

    if args.save:
        save_gen_pred(args, l_vqconfig, tag, pipeline, args.save_folder, unstd_pred,
                probs=probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--etag', type=str, required=True)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--sample_idx', type=int, default=None)
    parser.add_argument('-ut', '--use_text_transcriptions', action='store_true')
    parser.add_argument('-dsl', '--disable_strict_load', action='store_true')
    parser.add_argument('-ivp', '--input_video_path', type=str, required=True)
    parser.add_argument('-iap', '--input_audio_path', type=str, required=True)
    parser.add_argument('-itp', '--input_text_path', type=str, required=True)
    args = parser.parse_args()
    print(args)
    generate_prediction(args)
