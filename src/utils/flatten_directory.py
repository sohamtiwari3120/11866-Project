import argparse

import os
import shutil

def flatten_directory(args):
    os.makedirs(args.output_directory, exist_ok=True)
    ctr = 0
    for root, dirs, files in os.walk(args.input_directory):
        for filename in files:
            source_path = os.path.join(root, filename)
            destination_path = os.path.join(args.output_directory, filename)
            shutil.copy(source_path, destination_path)
            ctr += 1
    print(f'Copied {ctr} files from {args.input_directory} to {args.output_directory}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_directory', type=str, required=True)
    parser.add_argument('-od', '--output_directory', type=str, required=True)
    args = parser.parse_args()
    flatten_directory(args)