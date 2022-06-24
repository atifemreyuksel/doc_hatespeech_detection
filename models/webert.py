import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class WeBERT(nn.Module):
    def __init__(self, checkpoint, num_labels): 
        super(WeBERT, self).__init__() 
        self.num_labels = num_labels
        
        self.relu = nn.ReLU()
        #Load Model with given checkpoint and extract its body
        self.model = AutoModel.from_pretrained(checkpoint, config=AutoConfig.from_pretrained(checkpoint, output_attentions=True, output_hidden_states=True))
        self.dropout1 = nn.Dropout(0.2, inplace=False) 
        self.classifier1 = nn.Linear(768, 128) # load and initialize weights
        self.dropout2 = nn.Dropout(0.2, inplace=False) 
        self.classifier2 = nn.Linear(128, 2) # load and initialize weights
        
        self.weighter2 = nn.Linear(768 * 2, 2)
        self.w_dropout2 = nn.Dropout(0.2, inplace=False) 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        #Extract outputs from the body
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        sequence_output = self.dropout1(outputs[0]) #outputs[0]=last hidden state
        sequence_output = sequence_output[:, :2, :]
        weights = self.weighter2(sequence_output.contiguous().view(sequence_output.shape[0], -1))
        weights = self.w_dropout2(self.relu(weights))
        weights = torch.unsqueeze(self.softmax(weights), dim=2)
        
        sequence_output = torch.mean(sequence_output * weights, dim=1)
        sequence_output = torch.squeeze(sequence_output, 1)
        
        output = self.relu(self.classifier1(sequence_output)) # calculate losses
        output = self.dropout2(output)
        logits = self.classifier2(output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
