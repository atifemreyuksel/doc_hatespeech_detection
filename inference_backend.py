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


def load_model(load_from, is_gpu):
    checkpoint = torch.load(load_from, map_location=f'cpu')
    config_file = checkpoint['config_file']
    args = dotdict(json.load(open(config_file, 'r')))

    model = RuleBERT(checkpoint="dbmdz/bert-base-turkish-128k-uncased", num_labels=args.num_classes, rule_dimension=26)
    if is_gpu:
        device = 'cuda:0'
        model = nn.DataParallel(model)
    else:    
        device = f'cpu'
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

def detect_hate_speech(text, model, device):
    tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
    test_dataset = HateSpeechDataset(
        phase="inference",
        tokenizer=tokenizer,
        data_path=text,
        apply_preprocessing=True,
        add_ling_features=True
    )
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, rule, detected_feats in tqdm(test_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            rule = rule.to(device)
            
            test_output = model(input_ids=input_ids, attention_mask=attention_mask, rules=rule)
            logits = test_output.logits
            predictions = logits.argmax(dim=1).data
            detected_feats["prediction"] = predictions.cpu().numpy()[0]

    for key_type, detected_ones in detected_feats.items():
        if key_type == "prediction":
            detected_feats[key_type] = str(detected_ones)
        else:
            for detected_one in detected_ones:
                detected_one["span"] = [str(detected_one["span"][0].numpy()[0]), str(detected_one["span"][1].numpy()[0])]
                detected_one["match"] = detected_one["match"][0]
                detected_one["degree"] = detected_one["degree"][0]    
    return detected_feats
