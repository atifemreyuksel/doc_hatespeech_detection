import argparse
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_metric
from tqdm import tqdm
from transformers import AutoTokenizer

from models.model import HateSpeechModel

# Metrics
prec = load_metric("precision")
rec = load_metric("recall")
acc = load_metric("accuracy")
f1 = load_metric("f1")

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def is_bad_sentence(sentence):
    if len(sentence) > 10:
        return False
    else:
        return True

def filter_bad_sentences(sentences):
    filtered_sentences = []
    for sentence in sentences:
        # most sentences outside of this range are bad sentences
        if not is_bad_sentence(sentence):
            filtered_sentences.append(sentence)
    return filtered_sentences

def enforce_max_sent_per_example(sentences):
    if len(sentences) > args.max_sent_per_news:
        i = len(sentences) // 2
        l1 = enforce_max_sent_per_example(sentences[:i])
        l2 = enforce_max_sent_per_example(sentences[i:])
        return l1 + l2
    else:
        return [sentences]

def get_tokenized_sentence(sentence, is_first_sent=False):
    if is_first_sent:
        tokenized_sentence = tokenizer(sentence, truncation=True, padding='max_length', max_length=args.sent_max_len)['input_ids']
    else:
        tokenized_sentence = tokenizer(sentence, truncation=True, padding='max_length', max_length=args.sent_max_len+1)
        tokenized_sentence = {key: value[1:] for key, value in tokenized_sentence.items()}['input_ids']
    return tokenized_sentence

def text_to_instance(sentences):
    tokenized_sentences = [get_tokenized_sentence(s, is_first_sent=i==1) for i, s in enumerate(sentences)]

    if len(tokenized_sentences) < args.max_sent_per_news:
        padding_sentences = [get_tokenized_sentence("[PAD]", is_first_sent=False) for _ in range(args.max_sent_per_news - len(tokenized_sentences))]
        tokenized_sentences.extend(padding_sentences)
    return np.array(tokenized_sentences)
        

parser = argparse.ArgumentParser()
parser.add_argument('--load_from', type=str, default='', help='load the pretrained model from the specified location')
args = parser.parse_args()

gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = f'{gpu_id}'

checkpoint = torch.load(args.load_from, map_location=f'cuda:{gpu_id}')
config_file = checkpoint['config_file']
args = dotdict(json.load(open(config_file, 'r')))

is_multigpu = "0" in args.gpu_ids and "1" in args.gpu_ids

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")

data = pd.read_csv(args.dataset_dir, sep="|", converters={"sentences": pd.eval})
data = data[data["phase"] == "test"][["title", "sentences", "Label"]]
data["title"] = data["title"].apply(lambda title: title if isinstance(title, str) else "")
data["sentences"] = data.apply(lambda row: [row["title"]] + row["sentences"], axis=1)
data = data.drop("title", axis=1)
data["Label"] = data["Label"].map({"not_hate": 0, "hate": 1})


model = HateSpeechModel(emb_hidden_dim=100, gru_hidden_size=128, num_labels=args.num_classes)

if is_multigpu:
    device = 'cuda:0'
    model = nn.DataParallel(model)
else:    
    device = f'cuda:{args.gpu_ids}'
model.to(device)

model.load_state_dict(checkpoint['model_state_dict'])

gru_input = np.array(tokenizer("[PAD]", truncation=True, padding=True)['input_ids']).to(device)

test_results_df = pd.DataFrame(columns=["id", "label", "prediction", "all_predictions", "confidences", "importances"])

model.eval()
with torch.no_grad():
    for _, row in data.iterrows():
        sentences = row["sentences"]
        label = row["Label"]

        pred_labels, confidences, sentence_imp_scores = [], [], []

        sentences = filter_bad_sentences(sentences)
        for sentences_loop in enforce_max_sent_per_example(sentences):
            instance = text_to_instance(sentences=sentences_loop)

            instance = instance.to(device) 
            test_output = model(instance, gru_input)

            pred_idx, confidence = test_output['action_probs'].argmax(), test_output['action_probs'].max()
            importances = test_output['sentence_importances']
            label = pred_idx
            pred_labels.append(label)
            confidences.append("{:.2f}".format(confidence))
            sentence_imp_scores.append(importances)
        
        pred_label = Counter(pred_labels).most_common(1)[0][0]
        test_results_df = test_results_df.append({
                "id": row["id"],
                "label": label,
                "prediction": pred_label,
                "all_predictions": pred_labels,
                "confidences": confidences,
                "importances": sentence_imp_scores
            },
            ignore_index = True
        )
        f1.add_batch(predictions=pred_label, references=label)
        acc.add_batch(predictions=pred_label, references=label)
        rec.add_batch(predictions=pred_label, references=label)
        prec.add_batch(predictions=pred_label, references=label)

test_metrics = {
    "accuracy": acc.compute()['accuracy'],
    "precision": prec.compute()['precision'],
    "recall": rec.compute()['recall'],
    "f1": f1.compute()['f1']
}
print(test_metrics)

test_results_df.to_csv("test_results.csv", index=False)
