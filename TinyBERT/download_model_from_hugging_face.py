import argparse
import os

from transformers import AutoTokenizer, AutoModel

output_path = os.path.join('data', 'models')
os.makedirs(output_path, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', help='name of hugging face model', type=str)
    args = parser.parse_args()

    model_name = args.model_name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    out_dir = os.path.join(output_path, model_name)
    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_vocabulary(out_dir)
    model.save_pretrained(out_dir)
