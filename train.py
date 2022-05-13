import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AutoTokenizer

import utils

#from models.vggnet import Vggnet
from datasets.hatespeech_loader import HateSpeechDataset

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
    epoch_val_accuracy = 0


    model.eval()
    with torch.no_grad():
        epoch_val_accuracy = 0
        epoch_val_loss = 0

        preds, labels = [], []
        for data, label in val_loader:
            data = data.to(device)
            label = label.to(device)

            val_output = model(data)
            val_loss = criterion(val_output, label)

            acc = (val_output.argmax(dim=1) == label).float().mean()
            epoch_val_accuracy += acc / len(val_loader)
            epoch_val_loss += val_loss / len(val_loader)

            preds.extend(val_output.argmax(dim=1).data)
            labels.extend(label.data)
    val_metrics = {
        "accuracy": epoch_val_accuracy,
        "precision": precision_score(preds, labels),
        "recall": recall_score(preds, labels),
        "f1": f1_score(preds, labels)
    }
    return epoch_val_loss, val_metrics


parser = argparse.ArgumentParser()

# experiment specifics
parser.add_argument('--name', type=str, default='imagenet2mnist', help='name of the experiment. It decides where to store samples and models')        
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--seed', type=int, default=47)
parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

# training specifics       
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--num_workers', type=int, default=16, help='num workers for data prep')
parser.add_argument('--epochs', type=int, default=100, help='# of epochs in training')
parser.add_argument('--weight_decay', type=int, default=5e-4, help='Coefficient of weight decay for optimizer')

# for setting inputs
parser.add_argument('--dataset_dir', type=str, default='../data/data_cleaned_sentences_phases_2020-04-16.csv')  
parser.add_argument('--sent_max_len', type=int, default=200)  
parser.add_argument('--max_sent_per_news', type=int, default=10)  
parser.add_argument('--num_classes', type=int, required=True) 

# model and optimizer
parser.add_argument('--load_from', type=str, default='', help='load the pretrained model from the specified location')
parser.add_argument('--optimizer_type', type=str, default='adam', choices=["sgd", "adam", "rmsprop"], help='Name of the optimizer')
parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--lr_factor', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')

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
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
elif args.optimizer_type == "adam":
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.9), weight_decay=args.weight_decay)
else:
    raise NotImplementedError("choose adam or sgd")


init_epoch = 0
if args.load_from != "":
    checkpoint = torch.load(args.load_from)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    init_epoch = checkpoint['epoch']
    criterion = checkpoint['criterion']

best_loss, best_acc = np.inf, 0
for epoch in range(init_epoch, init_epoch + args.epochs):
    model, optimizer, epoch_accuracy, epoch_loss = train(model, train_loader, criterion, optimizer)
    epoch_val_loss, epoch_val_metrics = evaluation(model, val_loader, criterion)

    log_message = f"""Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} 
                                        - val_acc: {epoch_val_metrics['accuracy']:.4f} - val_precision: {epoch_val_metrics['precision']:.4f}
                                        - val_recall: {epoch_val_metrics['recall']:.4f} - val_f1_score: {epoch_val_metrics['f1']:.4f}\n"""
    print(log_message)
    with open(os.path.join(checkpoint_dir, f"training_log_{training_uid}.txt"), "a") as f:
        f.write(log_message)
    
    if epoch_val_loss < best_loss and epoch_val_metrics['accuracy'] > best_acc:
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'criterion': criterion,
                    'config_file': config_file
                }, 
                os.path.join(checkpoint_dir, f"epoch_{epoch}.pth")
            )
        best_loss = epoch_val_loss
        best_acc = epoch_val_metrics['accuracy']
