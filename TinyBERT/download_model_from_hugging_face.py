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

    tokenizer.save_vocabulary(os.path.join(output_path, model_name))
    model.save_pretrained(os.path.join(output_path, model_name))
