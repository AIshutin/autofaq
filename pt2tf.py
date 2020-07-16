import transformers
from transformers import TFAutoModel, AutoTokenizer, AutoModel
import os
import pickle
import tensorflow as tf
import embedlib
from embedlib.datasets import collate_wrapper
import torch
import tqdm

def convert2tf(model_dir):
    tokenizer = pickle.load(open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb'))
    # AutoTokenizer.from_pretrained('distilbert-base-uncased')
    #
    model = TFAutoModel.from_pretrained(os.path.join(model_dir, 'qembedder'), from_pt=True)

    text = "Hello, world!"
    tokenizer.encode(text, add_special_tokens=True, return_tensors='tf')

    return model, tokenizer

def save_model(tokenizer, model, model_dir):
    tokenizer.save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    # model._set_inputs(tf.TensorSpec([1, 384], tf.int32))
    # tf.saved_model.save(model, os.path.join(model_dir))

def main(pt_model_dir, tf_model_dir):
    model, tokenizer = convert2tf(pt_model_dir)
    save_model(tokenizer, model, tf_model_dir)

if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    torch.manual_seed(0)
    checkpoints = "checkpoints/"
    model_dir = os.path.join(checkpoints, os.listdir(checkpoints)[0])
    main(model_dir, 'distilbert')
