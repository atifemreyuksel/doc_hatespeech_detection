import argparse
import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer

from data_loaders.hatespeech_loader import HateSpeechDataset
from models.rulebert import RuleBERT

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__        

parser = argparse.ArgumentParser()
parser.add_argument('--load_from', type=str, default='', help='load the pretrained model from the specified location')
parser.add_argument('--text', type=str, default='', help='Article text')
parser.add_argument('--is_gpu', type=int, default=0, help='If there is gpu in the machine, please specify as 1')
args = parser.parse_args()

text = args.text
is_gpu = args.is_gpu
checkpoint = torch.load(args.load_from, map_location=f'cpu')
config_file = checkpoint['config_file']
args = dotdict(json.load(open(config_file, 'r')))

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
test_dataset = HateSpeechDataset(
    phase="inference",
    tokenizer=tokenizer,
    data_path=text,
    sent_max_len=args.sent_max_len,
    max_sent_per_news=args.max_sent_per_news,
    apply_preprocessing=True,
    add_ling_features=args.add_ling_features
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

model = RuleBERT(checkpoint="dbmdz/bert-base-turkish-128k-uncased", num_labels=args.num_classes, rule_dimension=26)

if is_gpu:
    device = 'cuda:0'
    model = nn.DataParallel(model)
else:    
    device = f'cpu'
model.to(device)

model.load_state_dict(checkpoint['model_state_dict'])

all_predictions = []
model.eval()
with torch.no_grad():
    for input_ids, attention_mask, rule in tqdm(test_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        rule = rule.to(device)
        
        test_output = model(input_ids=input_ids, attention_mask=attention_mask, rules=rule)
        logits = test_output.logits
        predictions = logits.argmax(dim=1).data

detected_feats = test_dataset.detected_patterns
detected_feats["prediction"] = str(predictions.cpu().numpy()[0])
print(text)
print(detected_feats)
