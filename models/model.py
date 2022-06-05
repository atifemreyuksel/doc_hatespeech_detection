import sys

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules import TextFieldEmbedder, TimeDistributed
from transformers import AutoConfig, AutoModel


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
        input_seq = self.embedder(inputs)
        # bert: 101 as CLS start token id for embedding
        input_mask = inputs['bert'] == 101
        input_seq = input_seq[input_mask]
        batch_size, _ = input_seq.size()
        
        # (1, batch size, input_size) add another dimension so that it works with the GRU
        embedded = input_seq.contiguous().view(batch_size, 1, self.input_size)  # B x S=1 x N
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
        elif len(inputs.shape) == 2:
            # HACK: Trick for using this layer for both sentence encoder and sentiment classifier parts.
            inputs = inputs.unsqueeze(1)
        self.reflection_inputs = torch.zeros_like(inputs, dtype=torch.int32)
        for i in range(self.num_classes):
            self.reflection_inputs[:, :, i] = i
        emb_matrix = self.emb_layer(self.reflection_inputs)
        weighted_cls_embeddings = torch.mean(emb_matrix * inputs[..., None], dim=2)
        return weighted_cls_embeddings

class ClassEmbedding2(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=300):
        super(ClassEmbedding2, self).__init__()
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
    def __init__(self, emb_hidden_dim=300, gru_hidden_size=128, num_labels=2):
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
        self.text_field_embedder = self.text_embedder.embeddings
        encoded_sentence_dim = self.text_embedder.embeddings.word_embeddings.embedding_dim

        self.td_ff_cls_emb_l1 = TimeDistributed(nn.Linear(encoded_sentence_dim, self.num_labels))
        self.cls_emb_layer = ClassEmbedding2(num_classes=self.num_labels, hidden_dim=self.emb_hidden_dim)
        # Document encoder
        gated_sentence_dim = encoded_sentence_dim + self.emb_hidden_dim
        self.td_ff_gate_l2 = TimeDistributed(nn.Linear(gated_sentence_dim, 1))

        self.gru_1 = nn.GRU(input_size=gated_sentence_dim, hidden_size=self.gru_hidden_size, batch_first=True,
                            bidirectional=False, dropout=0.0, num_layers=1)
        self.gru_decoder = DecoderGRU(embedder=self.text_field_embedder, input_size=encoded_sentence_dim,
                                      hidden_size=self.gru_hidden_size, output_size=self.gru_hidden_size)
        # Sentiment classifier
        self.ff_doc_emb = nn.Linear(self.gru_hidden_size, self.num_labels)
        self.ff_sent_cls = nn.Linear(self.gru_hidden_size + self.emb_hidden_dim, self.num_labels)

    def forward(self, sentences, gru_decoder_inputs, labels):
        # --------------- SENTENCE ENCODER --------------
        # embedded_sentences: batch_size, num_sentences, sentence_length, embedding_size
        embedded_sentences = self.text_field_embedder(sentences)
        batch_size, _, _, embedding_size = embedded_sentences.size()
        # The following code collects vectors of the SEP tokens from all the examples in the batch,
        # and arrange them in one list. It does the same for the labels and confidences.
        # Berturk: 3 as SEP token id for embedding
        sentences_mask = sentences['bert'] == 3  # mask for all the SEP tokens in the batch
        
        embedded_sentences = embedded_sentences[sentences_mask]  # BS x num_sentences_per_ex x sent_len x vector_len -> BS * max_sent_len x sent_len
        embedded_sentences = embedded_sentences.contiguous().view(batch_size, -1, embedding_size) # BS * max_sent_len x vector_len -> # BS x max_sentences x vector_len
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
        # Compute loss and epoch metrics
        output_dict = {"action_probs": label_probs}
        output_dict["sentence_importances"] = importance_coefs
        # =====================================================================

        if labels is not None:
            # Compute cross entropy loss
            flattened_labels = labels.contiguous().view(-1)
            label_loss = self.loss(label_logits, flattened_labels)
            output_dict["loss"] = label_loss
        
        output_dict['action_logits'] = label_logits
        return output_dict

class UFOModel(nn.Module):
    """
    Sentiment analysis model for news
    """

    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 bert_dropout: float = 0.1,
                 emb_hidden_dim=300,
                 gru_hidden_size=128
                 ) -> None:
        super(UFOModel, self).__init__(vocab)
        self.vocab = vocab
        self.emb_hidden_dim = emb_hidden_dim
        self.gru_hidden_size = gru_hidden_size
        self.num_labels = self.vocab.get_vocab_size(namespace='labels')

        self.dropout = torch.nn.Dropout(p=bert_dropout)
        self.relu = nn.ReLU()

        # define loss
        # TODO: Give class weights for class imbalance
        # [pos, neutral, neg] = [1., 1.24, 2.23]  
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        # define accuracy metrics
        self.label_accuracy = CategoricalAccuracy()
        self.label_f1_metrics = {}
        # define F1 metrics per label
        for label_index in range(self.num_labels):
            label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
            self.label_f1_metrics[label_name] = F1Measure(label_index)

        # Sentence encoder
        self.text_field_embedder = text_field_embedder
        encoded_sentence_dim = text_field_embedder._token_embedders['bert'].output_dim

        self.td_ff_cls_emb_l1 = TimeDistributed(Linear(encoded_sentence_dim, self.num_labels))
        self.cls_emb_layer = ClassEmbedding2(num_classes=self.num_labels, hidden_dim=self.emb_hidden_dim)
        # Document encoder
        gated_sentence_dim = encoded_sentence_dim + self.emb_hidden_dim
        self.td_ff_gate_l2 = TimeDistributed(Linear(gated_sentence_dim, 1))

        self.gru_1 = nn.GRU(input_size=gated_sentence_dim, hidden_size=self.gru_hidden_size, batch_first=True,
                            bidirectional=False, dropout=0.0, num_layers=1)
        self.gru_decoder = DecoderGRU(embedder=self.text_field_embedder, input_size=encoded_sentence_dim,
                                      hidden_size=self.gru_hidden_size, output_size=self.gru_hidden_size)
        # Sentiment classifier
        self.ff_doc_emb = nn.Linear(self.gru_hidden_size, self.num_labels)
        self.ff_sent_cls = nn.Linear(self.gru_hidden_size + self.emb_hidden_dim, self.num_labels)

    def forward(self,
                sentences: torch.LongTensor,
                gru_decoder_inputs: torch.LongTensor,
                labels: torch.IntTensor = None
                ) -> Dict[str, torch.Tensor]:
        # --------------- SENTENCE ENCODER --------------
        # embedded_sentences: batch_size, num_sentences, sentence_length, embedding_size
        embedded_sentences = self.text_field_embedder(sentences)
        batch_size, _, _, embedding_size = embedded_sentences.size()
        # The following code collects vectors of the SEP tokens from all the examples in the batch,
        # and arrange them in one list. It does the same for the labels and confidences.
        # scibert: 103, bert: 102 as SEP token id for embedding
        sentences_mask = sentences['bert'] == 102  # mask for all the SEP tokens in the batch
        
        embedded_sentences = embedded_sentences[sentences_mask]  # BS x num_sentences_per_ex x sent_len x vector_len -> BS * max_sent_len x sent_len
        embedded_sentences = embedded_sentences.contiguous().view(batch_size, -1, embedding_size) # BS * max_sent_len x vector_len -> # BS x max_sentences x vector_len
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
        # Compute loss and epoch metrics
        output_dict = {"action_probs": label_probs}
        output_dict["sentence_importances"] = importance_coefs
        # =====================================================================

        if labels is not None:
            # Compute cross entropy loss
            flattened_labels = labels.contiguous().view(-1)

            label_loss = self.loss(label_logits, flattened_labels)
            
            self.label_accuracy(label_probs.float().contiguous(), flattened_labels.squeeze(-1))

            # compute F1 per label
            for label_index in range(self.num_labels):
                label_name = self.vocab.get_token_from_index(namespace='labels', index=label_index)
                metric = self.label_f1_metrics[label_name]
                metric(label_probs, flattened_labels)
        
            output_dict["loss"] = label_loss
        
        output_dict['action_logits'] = label_logits
        return output_dict

