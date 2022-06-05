import argparse
import json
import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_metric
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from transformers import AutoTokenizer

import utils

#from models.vggnet import Vggnet
from data_loaders.hatespeech_loader import HateSpeechDataset

# Example command: python train.py --name test_mnist_eval_old --gpu_ids 1 --batch_size 32 --pretrained --dataset_dir data/MNIST --num_classes 10 --imsize 128 --eval_old_task

def train(model, train_loader, criterion, optimizer):
    epoch_loss = 0
    epoch_accuracy = 0

    model.train()
    for data, label in tqdm(train_loader):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        if utils.check_loss(loss, loss.item()):
            loss.backward()
            optimizer.step()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)
    return model, optimizer, epoch_accuracy, epoch_loss

def evaluation(model, val_loader, criterion):    
    epoch_val_loss = 0

    model.eval()
    with torch.no_grad():
        epoch_val_loss = 0

        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_loss += val_loss / len(val_loader)

            predictions = val_output.argmax(dim=1).data
            f1.add_batch(predictions=predictions, references=label.data)
            acc.add_batch(predictions=predictions, references=label.data)
            rec.add_batch(predictions=predictions, references=label.data)
            prec.add_batch(predictions=predictions, references=label.data)
    val_metrics = {
        "accuracy": acc.compute()['accuracy'],
        "precision": prec.compute()['precision'],
        "recall": rec.compute()['recall'],
        "f1": f1.compute()['f1']
    }
    return epoch_val_loss, val_metrics


parser = argparse.ArgumentParser()

# experiment specifics
parser.add_argument('--name', type=str, default='imagenet2mnist', help='name of the experiment. It decides where to store samples and models')        
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--seed', type=int, default=47)
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

# training specifics       
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--num_workers', type=int, default=16, help='num workers for data prep')
parser.add_argument('--epochs', type=int, default=4, help='# of epochs in training')
parser.add_argument('--steps_for_eval', type=int, default=500, help='# of steps for eval and saving model')

# for setting inputs
parser.add_argument('--dataset_dir', type=str, default='../data/data_cleaned_sentences_phases_2020-04-16.csv')  
parser.add_argument('--sent_max_len', type=int, default=200)  
parser.add_argument('--max_sent_per_news', type=int, default=30)  
parser.add_argument('--num_classes', type=int, required=True) 

# model and optimizer
parser.add_argument('--load_from', type=str, default='', help='load the pretrained model from the specified location')
parser.add_argument('--optimizer_type', type=str, default='adam', choices=["sgd", "adam", "adamw"], help='Name of the optimizer')
parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate for adam')

args = parser.parse_args()

checkpoint_dir = os.path.join(args.checkpoints_dir, args.name)
os.makedirs(checkpoint_dir, exist_ok=True)

training_uid = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3]

config_file = os.path.join(checkpoint_dir, f'config_{training_uid}.json')
json.dump(vars(args), open(config_file, 'w'))

utils.seed_everything(args.seed)
is_multigpu = "0" in args.gpu_ids and "1" in args.gpu_ids
num_new_classes = args.num_classes

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-128k-uncased")
train_dataset = HateSpeechDataset(phase="train", tokenizer=tokenizer, data_path=args.dataset_dir, sent_max_len=args.sent_max_len, max_sent_per_news=args.max_sent_per_news)
val_dataset = HateSpeechDataset(phase="val", tokenizer=tokenizer, data_path=args.dataset_dir, sent_max_len=args.sent_max_len, max_sent_per_news=args.max_sent_per_news)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers//2, pin_memory=True)

#model = Alexnet(pretrained=args.pretrained, num_new_classes=num_new_classes)

if is_multigpu:
    device = 'cuda:0'
    model = nn.DataParallel(model)
else:    
    device = f'cuda:{args.gpu_ids}'
model.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer selection
if args.optimizer_type == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
elif args.optimizer_type == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer_type == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
else:
    raise NotImplementedError("choose adam or sgd")

# Metrics
prec = load_metric("precision")
rec = load_metric("recall")
acc = load_metric("accuracy")
f1 = load_metric("f1")

lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, min_lr=1e-09, verbose=True)

init_epoch = 0
if args.load_from != "":
    checkpoint = torch.load(args.load_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    init_epoch = checkpoint['epoch']
    criterion = checkpoint['criterion']

best_f1 = 0
metric_df = pd.DataFrame(columns=["epoch", "step", "F1", "Accuracy", "Precision", "Recall"])

for epoch in range(init_epoch, init_epoch + args.epochs):
    epoch_loss = 0
    epoch_accuracy = 0

    model.train()
    for i, data, label in enumerate(tqdm(train_loader)):
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label)

        if utils.check_loss(loss, loss.item()):
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        acc = (output.argmax(dim=1) == label).float().mean()
        epoch_accuracy += acc / len(train_loader)
        epoch_loss += loss / len(train_loader)

        if i % args.steps_for_eval == 0 and i != 0:
            model.eval()

            with torch.no_grad():
                epoch_val_loss = 0

                for data, label in val_loader:
                    data = data.to(device)
                    label = label.to(device)

                    val_output = model(data)
                    val_loss = criterion(val_output, label)

                    acc = (val_output.argmax(dim=1) == label).float().mean()
                    epoch_val_loss += val_loss / len(val_loader)

                    predictions = val_output.argmax(dim=1).data
                    f1.add_batch(predictions=predictions, references=label.data)
                    acc.add_batch(predictions=predictions, references=label.data)
                    rec.add_batch(predictions=predictions, references=label.data)
                    prec.add_batch(predictions=predictions, references=label.data)
            
            val_metrics = {
                "accuracy": acc.compute()['accuracy'],
                "precision": prec.compute()['precision'],
                "recall": rec.compute()['recall'],
                "f1": f1.compute()['f1']
            }

            if val_metrics['f1'] > best_f1:
                torch.save({
                            'epoch': epoch,
                            'steps': i,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'criterion': criterion,
                            'config_file': config_file
                        }, 
                        os.path.join(checkpoint_dir, f"best_model.pth")
                    )
                best_f1 = val_metrics['f1']            
            
            lr_scheduler.step(val_metrics['f1'])
            
            model.train()

            log_message = f"""Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} 
                                                - val_acc: {val_metrics['accuracy']:.4f} - val_precision: {val_metrics['precision']:.4f}
                                                - val_recall: {val_metrics['recall']:.4f} - val_f1_score: {val_metrics['f1']:.4f}\n"""

            with open(os.path.join(checkpoint_dir, f"training_log_{training_uid}.txt"), "a") as f:
                f.write(log_message)
            metric_df =  metric_df.append({
                    "epoch": epoch,
                    "step": i,
                    "Precision": val_metrics['precision'],
                    "Recall": val_metrics['recall'],
                    "Accuracy": val_metrics['accuracy'],
                    "F1": val_metrics['f1']
                },
                ignore_index = True
            )
            metric_df.to_csv(os.path.join(checkpoint_dir, "metrics_df.csv"), index=False)
metric_df.to_csv(os.path.join(checkpoint_dir, "metrics_df.csv"), index=False)
