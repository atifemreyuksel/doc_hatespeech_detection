import sys

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        y = self.module(x_reshape)
        # We have to reshape Y
        y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        return y

class ContextAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ContextAttention, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, decoder_hidden, encoder_outputs):
        batch_size = decoder_hidden.size(0)
        # For the dot scoring method, no weights or linear layers are involved
        alignment_scores = encoder_outputs.bmm(decoder_hidden.contiguous().view(batch_size, self.hidden_size, 1))
        attn_weights = F.softmax(alignment_scores, dim=2).transpose(1, 2)
        return attn_weights
        
class DecoderGRU(nn.Module):
    def __init__(self, embedder, input_size, hidden_size, output_size,
                 n_layers=1, dropout=0.1):
        super(DecoderGRU, self).__init__()

        # Keep for reference
        self.embedder = embedder
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.gru = nn.GRU(self.input_size, self.hidden_size, n_layers, dropout=self.dropout, batch_first=True)
        self.attention = ContextAttention(self.hidden_size)
        self.concat = nn.Linear(self.hidden_size * 2, self.output_size)

    def forward(self, inputs, last_hidden, encoder_outputs):
        """
        input_seq : batch_size, hidden_size
        hidden : hidden_size, batch_size
        encoder_outputs : batch_size, max input length, hidden_size
        """
        input_seq = self.embedder(input_ids=inputs)
        # 0: CLS token id in Huggingface models
        embedded = input_seq[:, 0, :]#.contiguous().view(input_seq.size()[0], -1)
        batch_size, _ = embedded.size()
        
        embedded = embedded.contiguous().view(batch_size, 1, self.input_size)  # B x S=1 x N
        # Get current hidden state from input word and last hidden state
        rnn_output, _ = self.gru(embedded, last_hidden)
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs to get weighted average
        attn_weights = self.attention(rnn_output, encoder_outputs) # B, 1, max input length

        # (batch_size, 1, max input length) @ batch_size, max input length, hidden size
        # note that we use this convention here to take advantage of the bmm function
        context = attn_weights.bmm(encoder_outputs)  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(1)  # B x S=1 x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        output = torch.tanh(self.concat(concat_input))
        return output

class ClassEmbedding(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=300):
        super(ClassEmbedding, self).__init__()
        self.num_classes = num_classes
        self.emb_layer = nn.Embedding(num_embeddings=num_classes, embedding_dim=hidden_dim)
        
    def forward(self, inputs):
        if len(inputs.shape) < 2:
            raise Exception('# of dimensions of the given input must be at least 2')
        emb_matrix = einops.repeat(self.emb_layer.weight, 'm n -> k m n', k=inputs.shape[0])
        if len(inputs.shape) == 2:
            inputs = inputs.unsqueeze(1)
        weighted_cls_embeddings = torch.bmm(inputs, emb_matrix)
        return weighted_cls_embeddings

class HateSpeechModel(nn.Module):
    """Hate speech detection model for Turkish news"""
    def __init__(self, emb_hidden_dim=100, gru_hidden_size=128, num_labels=2):
        super(HateSpeechModel, self).__init__()
        
        self.emb_hidden_dim = emb_hidden_dim
        self.gru_hidden_size = gru_hidden_size
        self.num_labels = num_labels

        self.dropout = nn.Dropout(0.1, inplace=False)
        self.relu = nn.ReLU()

        self.loss = nn.CrossEntropyLoss()
        
        # Sentence encoder
        self.text_embedder = AutoModel.from_pretrained(
            "dbmdz/bert-base-turkish-128k-uncased",
            config=AutoConfig.from_pretrained(
                "dbmdz/bert-base-turkish-128k-uncased",
                output_attentions=True,
                output_hidden_states=True
            )
        )
        #self.text_field_embedder = self.text_embedder.embeddings        
        self.encoded_sentence_dim = self.text_embedder.embeddings.word_embeddings.embedding_dim

        self.td_ff_cls_emb_l1 = TimeDistributed(nn.Linear(self.encoded_sentence_dim, self.num_labels))
        self.cls_emb_layer = ClassEmbedding(num_classes=self.num_labels, hidden_dim=self.emb_hidden_dim)
        # Document encoder
        gated_sentence_dim = self.encoded_sentence_dim + self.emb_hidden_dim
        self.td_ff_gate_l2 = TimeDistributed(nn.Linear(gated_sentence_dim, 1))

        self.gru_1 = nn.GRU(input_size=gated_sentence_dim, hidden_size=self.gru_hidden_size, batch_first=True,
                            bidirectional=False, dropout=0.0, num_layers=1)
        self.gru_decoder = DecoderGRU(embedder=self.text_embedder.embeddings, input_size=self.encoded_sentence_dim,
                                      hidden_size=self.gru_hidden_size, output_size=self.gru_hidden_size)
        # Sentiment classifier
        self.ff_doc_emb = nn.Linear(self.gru_hidden_size, self.num_labels)
        self.ff_sent_cls = nn.Linear(self.gru_hidden_size + self.emb_hidden_dim, self.num_labels)

    def forward(self, input_ids, attention_mask, gru_decoder_inputs):
        # --------------- SENTENCE ENCODER --------------
        ### SEP token TRY
        #embedded_sentences = torch.zeros((sentences.shape[0], sentences.shape[1], self.encoded_sentence_dim))
        #for i in range(sentences.shape[0]):
        #    sentence_i = torch.squeeze(sentences[i, ...], 0)
        #    embedded_sentence_i = self.text_field_embedder(input_ids=sentence_i)

        #    sep_idxs = torch.where(sentence_i == 3)[1].cpu().numpy()
        #    sep_sentence_i = torch.zeros((sentence_i.shape[0], self.encoded_sentence_dim))
        #    for j in range(sentences.shape[1]):
        #        sep_sentence_i[j, :] = embedded_sentence_i[j, sep_idxs[j], :]
        #    embedded_sentences[i, :, :] = sep_sentence_i    
        #embedded_sentences = embedded_sentences.to('cuda:0')
        
        ### Averaging embeddings TRY
        #batch_size, nof_sentences, _ = sentences.size()
        #sentences = einops.rearrange(sentences, 'b s t -> (b s) t')
        #embedded_sentences = self.text_field_embedder(input_ids=sentences)
        #embedded_sentences = einops.rearrange(embedded_sentences, '(b s) t h -> b s t h', b=batch_size, s=nof_sentences)
        #embedded_sentences = torch.mean(embedded_sentences, dim=2)

        #print(input_ids.size())        
        embedded_sentences = self.text_embedder(input_ids=input_ids, attention_mask=attention_mask)
        embedded_sentences = embedded_sentences[0][:, :20, :]
        #print(embedded_sentences.size())
        

        class_sim_embeddings = self.td_ff_cls_emb_l1(embedded_sentences) # BS x max_sentences x vector_len -> BS x max_sentences x # of classes
        class_sim_embeddings = self.relu(class_sim_embeddings)
        class_sim_embeddings = self.cls_emb_layer(class_sim_embeddings)
        embedded_sentences = torch.cat((embedded_sentences, class_sim_embeddings), axis=2)

        # --------------- DOCUMENT ENCODER ------------------
        gated_embedded_sentences = self.td_ff_gate_l2(embedded_sentences) # -> BS x max_sentences x 1
        importance_coefs = torch.sigmoid(gated_embedded_sentences)
        gated_embedded_sentences = importance_coefs * embedded_sentences  # -> BS x max_sentences x (vector_len + # of classes)

        encoder_outputs, last_hs = self.gru_1(gated_embedded_sentences) # -> BS x max_sentences x gru_hidden_size | 1 x BS, gru_hidden_size
        document_embeddings = self.gru_decoder(gru_decoder_inputs, last_hs, encoder_outputs)
        # --------------- SENTIMENT CLASSIFIER --------------
        class_doc_embeddings = self.ff_doc_emb(document_embeddings) # BS x vector_len -> BS x # of classes
        class_doc_embeddings = self.relu(class_doc_embeddings)
        class_doc_embeddings = self.cls_emb_layer(class_doc_embeddings).squeeze(1)
        document_cls_embeddings = torch.cat((document_embeddings, class_doc_embeddings), axis=1)

        label_logits = self.ff_sent_cls(document_cls_embeddings)
        label_probs = F.softmax(label_logits, dim=-1)

        # Create output dictionary for the trainer
        output_dict = {
            "action_logits": label_logits,
            "action_probs": label_probs,
            "sentence_importances": importance_coefs
        }
        return output_dict
