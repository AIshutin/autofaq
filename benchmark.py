import transformers
from transformers import TFAutoModel, BertTokenizer, AutoModel
import os
import pickle
import tensorflow as tf
import embedlib
from embedlib.datasets import collate_wrapper
import torch
import tqdm

def normalize(a, axis=1): # checked
    return a / tf.norm(a, axis=axis, keepdims=True)

def cosine_similarity_table(X, Y):
    X = normalize(X)
    Y = normalize(Y)
    return tf.matmul(X, tf.transpose(Y))

def run_tf_model(model, tensor):
    return tf.reduce_sum(model(tensor)[0], 1)

def process(model, tokenizer, text):
    tensor = tokenizer.encode(text, add_special_tokens=True, return_tensors='tf')
    return run_tf_model(model, tensor)

def calc_mrr(X, Y):
    csim = cosine_similarity_table(X, Y)
    # print('csim', csim)
    value = 0
    for i in range(csim.shape[0]):
        pos = 1
        for j in range(csim.shape[0]):
            if csim[i][j] > csim[i][i]:
                pos += 1
        value += 1 / pos
    return value, X.shape[0]

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, input_word_ids):
        input_mask = tf.ones(input_word_ids.shape)
        input_type_ids = tf.zeros(input_word_ids.shape)
        return self.model([{'input_ids': input_word_ids}],
                          training=False)

def load_model(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    reloaded = tf.keras.models.load_model(os.path.join(model_dir))
    return ModelWrapper(reloaded), tokenizer

if __name__ == "__main__":
    from torch.utils.data import Dataset, DataLoader
    torch.manual_seed(0)

    model, tokenizer = load_model('distilbert')

    corpus = embedlib.datasets.CorpusData(['en-twitt-corpus'], int(1e4))
    print("Corpus size: " + str(len(corpus)))
    test_split = 0.2
    test_size = int(len(corpus) * test_split)
    train_size = len(corpus) - test_size
    _, test_data = torch.utils.data.random_split(corpus, [train_size, test_size])
    test_sampler = None
    shuffle = True

    test_loader = DataLoader(test_data, batch_size=16, shuffle=shuffle, collate_fn=collate_wrapper)
    total = 0
    good = 0
    for batch in tqdm.tqdm(test_loader):
        questions, answers = batch.quests, batch.answs
        questions = [process(model, tokenizer, el) for el in questions]
        answers = [process(model, tokenizer, el) for el in answers]

        questions = tf.concat(questions, axis=0)
        # print('questions', questions)
        answers = tf.concat(answers, axis=0)
        # print('answers', answers)
        cgood, ctotal = calc_mrr(questions, answers)
        total += ctotal
        good += cgood

    print(f"{good} out of {total}")
    print(f"{good/total:4.4f}")
