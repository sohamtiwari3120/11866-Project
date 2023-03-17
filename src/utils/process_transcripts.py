import argparse
import os
import numpy as np

from load_utils import load_transcripts
from sentence_transformers import SentenceTransformer

def generate_sentence_embeddings(sentences, model_name):
    """Function to generate the sentence embeddings for a list of sentences using the desired model name

    Args:
        sentences (List[str]): List of sentence strings
        model_name (str): Huggingface model checkpoint name

    Returns:
        np.ndarray: numpy array containing embeddings of all the sentences in the input list
    """    
    # Load the pre-trained sentence transformer model
    model = SentenceTransformer(model_name)
    # Generate embeddings for the input sentences
    embeddings = model.encode(sentences)
    return embeddings


def generate_transcript_embeddings(dir_path, model_name):
    transcripts_dict = load_transcripts(dir_path)
    embeddings = generate_sentence_embeddings([obj['full_text'] for obj in list(transcripts_dict.values())], model_name)
    return transcripts_dict, embeddings

def main(dir_path, model_name, output_dir):
    if output_dir == "":
        output_dir = os.path.join(os.path.dirname(os.path.abspath(dir_path)), "transcript_embeddings", os.path.basename(model_name))
    os.makedirs(output_dir, exist_ok=True)

    transcripts_dict, embeddings = generate_transcript_embeddings(dir_path, model_name)
    
    for i, key in enumerate(transcripts_dict.keys()):
        with open(os.path.join(output_dir, f"{key}.npy"), "wb") as f:
            np.save(f, embeddings[i])
    print(f"Saved the processed transcripts in {os.path.join(output_dir)}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process transcript text files')
    parser.add_argument('-dp', '--dir_path', metavar='dir_path', type=str,
                        help='path to directory containing transcript text files')
    parser.add_argument('-mn', '--model_name', type=str, help='Sentence Transformer model name to be used for generating the sentence embeddings', default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument('-od', '--output_dir', type=str, help='Path to directory where you want the transcript texts and embeddings want to be saved', default="")
    args = parser.parse_args()
    main(args.dir_path, args.model_name, args.output_dir)