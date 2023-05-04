import argparse
import pickle
import numpy as np
import os

def main(args):
    pickle_folder_path = args.pickle_folder_path
    output_path = args.output_path
    window_length = args.window_length

    pickle_files = [os.path.join(pickle_folder_path, f) for f in os.listdir(pickle_folder_path)]
    data = []
    num_of_windows = len(pickle_files) // window_length
    for i in range(num_of_windows):
        temp = []
        for j in range(window_length):
            with open(pickle_files[i*window_length+j], 'rb') as f:
                pkl_data = pickle.load(f)
                expr_data = pkl_data['exp'].cpu().detach().numpy().squeeze()
                pose_data = pkl_data['pose'].cpu().detach().numpy().squeeze()
                temp.append(np.concatenate([expr_data, pose_data], axis=0))
        data.append(np.array(temp))
    data_np = np.array(data)
    np.save(output_path, data_np)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_folder_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--window_length', type=int, default=64)
    args = parser.parse_args()
    main(args)
