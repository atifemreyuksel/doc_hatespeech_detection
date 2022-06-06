from os import truncate

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm


class HateSpeechDataset(Dataset):
    def __init__(
        self,
        phase,
        tokenizer,
        data_path="../data/data_cleaned_sentences_phases_2020-04-16.csv",
        sent_max_len: int = 200,
        max_sent_per_news: int = 30,
    ):
        self.label_encodings = {"not_hate": 0, "hate": 1}
        self.rev_label_encodings = {0: "not_hate", 1: "hate"}
        self.tokenizer = tokenizer
        self.sent_max_len = sent_max_len
        self.max_sent_per_news = max_sent_per_news
        self.instances, self.labels = [], []
        self.__process_dataset(data_path, phase)
        print(len(self.instances), len(self.labels))

        self.gru_token = np.array(self.tokenizer("[PAD]", truncation=True, padding=True)['input_ids'])

    def __len__(self):
        return len(self.instances)

    def _read_phase_data(self, data_path, phase):
        data = pd.read_csv(data_path, sep="|", converters={"sentences": pd.eval})
        data = data[data["phase"] == phase][["title", "sentences", "Label"]]
        data["title"] = data["title"].apply(lambda title: title if isinstance(title, str) else "")
        data["sentences"] = data.apply(lambda row: [row["title"]] + row["sentences"], axis=1)
        data = data.drop("title", axis=1)
        data["Label"] = data["Label"].map(self.label_encodings)
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
        if len(sentences) > self.max_sent_per_news and self.max_sent_per_news > 0:
            i = len(sentences) // 2
            l1 = self._enforce_max_sent_per_example(sentences[:i])
            l2 = self._enforce_max_sent_per_example(sentences[i:])
            return l1 + l2
        else:
            return [sentences]

    def __get_tokenized_sentence(self, sentence, is_first_sent=False):
        if is_first_sent:
            tokenized_sentence = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.sent_max_len)['input_ids']
        else:
            tokenized_sentence = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=self.sent_max_len+1)
            tokenized_sentence = {key: value[1:] for key, value in tokenized_sentence.items()}['input_ids']
        return tokenized_sentence

    def __text_to_instance(self, sentences):
        tokenized_sentences = [self.__get_tokenized_sentence(s, is_first_sent=i==1) for i, s in enumerate(sentences)]

        if len(tokenized_sentences) < self.max_sent_per_news:
            padding_sentences = [self.__get_tokenized_sentence("[PAD]", is_first_sent=False) for _ in range(self.max_sent_per_news - len(tokenized_sentences))]
            tokenized_sentences.extend(padding_sentences)
        return tokenized_sentences

    def _process_one_news(self, sentences, label):
        sentences = self.__filter_bad_sentences(sentences)

        if len(sentences) == 0:
            return [], []

        instances, labels = [], []
        for sentences_loop in self._enforce_max_sent_per_example(sentences):
            instances_loop = self.__text_to_instance(sentences=sentences_loop)
            instances.append(instances_loop)
            labels.append(label)
        return instances, labels

    def __process_dataset(self, data_path, phase):
        data = self._read_phase_data(data_path, phase)
        for _, row in tqdm(data.iterrows()):
            row_instances, row_labels = self._process_one_news(row["sentences"], row["Label"])
            self.instances.extend(row_instances)
            self.labels.extend(row_labels)

    def __getitem__(self, idx):
        sentences, label = self.instances[idx], self.labels[idx]
        return np.array(sentences), label, self.gru_token
