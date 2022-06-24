import argparse
import json
import os

import torch
import torch.nn as nn
from datasets import load_metric
from tqdm import tqdm
from transformers import AutoTokenizer

from data_loaders.hatespeech_loader import HateSpeechDataset
from models.attwebert import AttWeBERT
from models.rulebert import RuleBERT
from models.webert import WeBERT

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
test_dataset = HateSpeechDataset(
    phase="test",
    tokenizer=tokenizer,
    data_path=args.dataset_dir,
    sent_max_len=args.sent_max_len,
    max_sent_per_news=args.max_sent_per_news,
    apply_preprocessing=args.apply_preprocessing,
    add_ling_features=args.add_ling_features
)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

if args.model_type == "attwebert":
    model = AttWeBERT(emb_hidden_dim=100, gru_hidden_size=128, num_labels=args.num_classes)
elif args.model_type == "webert":
    model = WeBERT(checkpoint="dbmdz/bert-base-turkish-128k-uncased", num_labels=args.num_classes)
elif args.model_type == "rulebert":
    model = RuleBERT(checkpoint="dbmdz/bert-base-turkish-128k-uncased", num_labels=args.num_classes, rule_dimension=26)

if is_multigpu:
    device = 'cuda:0'
    model = nn.DataParallel(model)
else:    
    device = f'cuda:{args.gpu_ids}'
model.to(device)

model.load_state_dict(checkpoint['model_state_dict'])

all_predictions = []
model.eval()
with torch.no_grad():
    for input_ids, attention_mask, label, gru_input in tqdm(test_loader):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        if args.add_ling_features:
            rule = gru_input.to(device)
        else:
            gru_input = gru_input.to(device)
        
        if args.model_type == "attwebert":
            test_output = model(input_ids, attention_mask, gru_input)
            logits = test_output['action_logits']
        elif args.model_type == "webert":
            test_output = model(input_ids, attention_mask, label)
            logits = test_output.logits
        elif args.model_type == "rulebert":
            test_output = model(input_ids, attention_mask, label, rule)
            logits = test_output.logits

        predictions = logits.argmax(dim=1).data
        f1.add_batch(predictions=predictions, references=label.data)
        acc.add_batch(predictions=predictions, references=label.data)
        rec.add_batch(predictions=predictions, references=label.data)
        prec.add_batch(predictions=predictions, references=label.data)
        all_predictions.extend(predictions.cpu().numpy())
test_metrics = {
    "accuracy": acc.compute()['accuracy'],
    "precision": prec.compute()['precision'],
    "recall": rec.compute()['recall'],
    "f1": f1.compute()['f1']
}
print(test_metrics)
with open(os.path.join("checkpoints/4_multihead_select", "test_metrics.json"), "w") as f:
    json.dump(test_metrics, f)
pred_df = test_dataset._get_prediction_results(all_predictions)
pred_df.to_csv(os.path.join("checkpoints/4_multihead_select", "test_predictions.csv"), index=False)
