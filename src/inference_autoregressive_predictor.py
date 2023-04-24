from test_autoregressive_predictor import *
import pandas as pd

def extract_samples(test_X, test_Y, test_audio, test_transcript_embs, test_files, path_list=[]):
    if len(path_list) == 0:
        return test_X, test_Y, test_audio, test_transcript_embs, test_files
    test_X_new, test_Y_new, test_audio_new, test_transcript_embs_new, test_files_new = [], [], [], [], []
    for i in range(test_files.shape[0]):
        if test_files[i, 0, 0] in path_list:
            test_X_new.append(test_X[i])
            test_Y_new.append(test_Y[i])
            test_audio_new.append(test_audio[i])
            test_transcript_embs_new.append(test_transcript_embs[i])
            test_files_new.append(test_files[i])
    return np.array(test_X_new), np.array(test_Y_new), np.array(test_audio_new), np.array(test_transcript_embs_new), np.array(test_files_new)

def compute_inference_l2(args):
    input_csv_file_df = pd.read_csv(args.input_file_path)
    input_csv_grouped = input_csv_file_df.groupby('listener')
    listener_video_dict = {listener: values['path'].to_list() for listener, values in input_csv_grouped}
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
    l_vq_model, _, _ = setup_vq_transformer(args, l_vqconfig,
                                            load_path=l_model_path,
                                            test=True)
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
    print(f"Results for paths provided in {args.input_file_path}")
    for listener in listener_video_dict:
        out_num = 1 if listener == 'fallon' else 0
        test_X, test_Y, test_audio, test_transcript_embs, test_files, _ = \
                load_test_data(config, pipeline, tag, out_num=out_num,
                            vqconfigs=vq_configs, smooth=True,
                            speaker=listener, num_out=num_out)
        
        test_X, test_Y, test_audio, test_transcript_embs, test_files = \
                extract_samples(test_X, test_Y, test_audio, test_transcript_embs, test_files, path_list=listener_video_dict[listener])

        ## run model and save/eval
        unstd_pred, probs, unstd_ub = run_model(args, config, l_vq_model, autoregressive_generator,
                                                test_X, test_Y, test_audio, test_transcript_embs, seq_len,
                                                patch_size, rng=rng)
        subset = (test_X.shape[0] // config['batch_size']) * config['batch_size']
        overall_l2 = np.mean(
            np.linalg.norm(test_Y[:subset,seq_len:,:] - unstd_pred[:,seq_len:,:], axis=-1))
        print(f"Listener - {listener}; description - {args.etag}\noverall l2: {overall_l2}\n{'-'*5}")
        if args.save:
            save_pred(args, l_vqconfig, tag, pipeline, test_files, unstd_pred,
                    probs=probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--etag', type=str, required=True)
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--sample_idx', type=int, default=None)
    parser.add_argument('-ut', '--use_text_transcriptions', action='store_true')
    parser.add_argument('-dsl', '--disable_strict_load', action='store_true')
    parser.add_argument('-ifp', '--input_file_path', type=str, required=True)
    args = parser.parse_args()
    print(args)
    compute_inference_l2(args)
