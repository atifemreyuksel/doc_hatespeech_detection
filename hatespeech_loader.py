import itertools
import os

import pandas as pd
from torch.utils.data import Dataset


class HateSpeechDataset(Dataset):
    def __init__(
        self,
        phase,
        tokenizer,
        data_path="../data/data_cleaned_sentences_phases_2020-04-16.csv",
        sent_max_len: int = 100,
        max_sent_per_news: int = 20,
    ):
        self.tokenizer = tokenizer
        self.sent_max_len = sent_max_len
        self.max_sent_per_news = max_sent_per_news
        self.instances, self.labels = self.__process_dataset(data_path, phase)

    def __len__(self):
        return len(self.instances)

    def __read_phase_data(self, data_path, phase):
        data = pd.read_csv(data_path, sep="|", converters={"sentences": pd.eval})
        data = data[data["phase"] == phase][["title", "sentences", "Label"]]
        data["title"] = data["title"].apply(lambda title: title if isinstance(title, str) else "")
        data["sentences"] = data.apply(lambda row: [row["title"]] + row["sentences"], axis=1)
        data = data.drop("title", axis=1)
        return data

    def __is_bad_sentence(self, sentence):
        if len(sentence) > 10:
            return False
        else:
            return True

    def __filter_bad_sentences(self, sentences):
        filtered_sentences = []
        for sentence in sentences:
            # most sentences outside of this range are bad sentences
            if not self.__is_bad_sentence(sentence):
                filtered_sentences.append(sentence)
        return filtered_sentences

    def _enforce_max_sent_per_example(self, sentences):
        """
        Splits examples with len(sentences) > self.max_sent_per_example into multiple smaller examples
        with len(sentences) <= self.max_sent_per_example.
        Recursively split the list of sentences into two halves until each half
        has len(sentences) < <= self.max_sent_per_example. The goal is to produce splits that are of almost
        equal size to avoid the scenario where all splits are of size
        self.max_sent_per_example then the last split is 1 or 2 sentences
        This will result into losing context around the edges of each examples.
        """
        if len(sentences) > self.max_sent_per_example and self.max_sent_per_example > 0:
            i = len(sentences) // 2
            l1 = self.enforce_max_sent_per_example(sentences[:i])
            l2 = self.enforce_max_sent_per_example(sentences[i:])
            return l1 + l2
        else:
            return [sentences]

    def _text_to_instance(self, sentences):
        tokenized_sentences = [
            self._tokenizer.tokenize(s)[: self.sent_max_len] + [Token("[SEP]")] for s in sentences
        ]

        # With padding, # of sentences in all instance will be the same for model.
        if len(tokenized_sentences) < self.max_sent_per_example:
            padding_sentences = [
                [Token("[PAD]"), Token("[SEP]")]
                for i in range(self.max_sent_per_example - len(tokenized_sentences))
            ]
            tokenized_sentences.extend(padding_sentences)
        sentences = [list(itertools.chain.from_iterable(tokenized_sentences))[:-1]]

        # fields["gru_decoder_inputs"] = TextField([Token("[PAD]")], self._token_indexers)

        return sentences

    def _process_one_news(self, sentences, label):
        sentences = self.__filter_bad_sentences(sentences)

        if len(sentences) == 0:
            return [], []

        instances, labels = [], []
        for sentences_loop in self._enforce_max_sent_per_example(sentences):
            instances_loop = self.text_to_instance(sentences=sentences_loop)
            instances.append(instances_loop)
            labels.append(label)
        return instances, labels

    def process_dataset(self, data_path, phase):
        data = self.__read_phase_data(data_path, phase)
        for _, row in data.iterrows():
            row_instances, row_labels = self._process_one_news(row["sentences"], row["Label"])
            self.instances.extend(row_instances)
            self.labels.extend(row_labels)

    def __getitem__(self, idx):
        sentences, label = self.instances[idx], self.labels[idx]
        return sentences, label
