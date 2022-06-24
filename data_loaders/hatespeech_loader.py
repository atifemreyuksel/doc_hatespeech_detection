import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data_loaders.data_cleaner import Cleaner
from data_loaders.ling_feat_generator import LinguisticRuleGenerator


class HateSpeechDataset(Dataset):
    def __init__(
        self,
        phase,
        tokenizer,
        data_path="../data/data_cleaned_sentences_phases_2022-04-16.csv",
        sent_max_len: int = 200,
        max_sent_per_news: int = 30,
        apply_preprocessing=False,
        add_ling_features=False,
    ):
        self.label_encodings = {"not_hate": 0, "hate": 1}
        self.rev_label_encodings = {0: "not_hate", 1: "hate"}
        self.tokenizer = tokenizer
        self.sent_max_len = sent_max_len
        self.max_sent_per_news = max_sent_per_news
        self.apply_preprocessing = apply_preprocessing
        self.add_ling_features = add_ling_features

        self.__process_dataset(data_path, phase)
        print(len(self.input_ids), len(self.attention_masks), len(self.labels))

        self.gru_token = np.array(self.tokenizer("[PAD]", truncation=True, padding=True)['input_ids'])
        
    def __len__(self):
        return len(self.input_ids)

    def _read_phase_data(self, data_path, phase):
        data = pd.read_csv(data_path, sep="|", converters={"sentences": pd.eval})
        data = data.sample(400)
        data = data[data["phase"] == phase]
        if self.apply_preprocessing:
            text_cleaner = Cleaner()
            data = text_cleaner.process_df(data)
        else:
            data = data[["id", "title", "sentences", "Label"]]
            data["title"] = data["title"].apply(lambda title: title if isinstance(title, str) else "")
            data["text"] = data.apply(lambda row: " ".join([sent for sent in [row["title"]] + row["sentences"]]), axis=1)
        if self.add_ling_features:
            rule_assigner = LinguisticRuleGenerator()
            data =  rule_assigner.apply_rules(data)
            data["all_rules"] = data.apply(lambda row: np.array(row["special_pattern"] + [row["general_rule"]] + row["anti_hs"] + row["hs_specific_verb"] + row["adj_bef_keyword"] + row["adj_after_keyword"]).astype(np.float32), axis=1)
            data = data.drop(["special_pattern", "general_rule", "anti_hs", "hs_specific_verb", "adj_bef_keyword", "adj_after_keyword"], axis=1) 
        data = data.drop("title", axis=1)
        data["Label"] = data["Label"].map(self.label_encodings)
        return data

    def __process_dataset(self, data_path, phase):
        data = self._read_phase_data(data_path, phase)
        texts = list(data["text"].values)
        self.labels = list(data["Label"].values)
        instances = self.tokenizer(texts, truncation=True, padding=True)
        self.input_ids = instances['input_ids']
        self.attention_masks = instances['attention_mask']
        self.idxs = list(data["id"].values)
        if self.add_ling_features:
            self.rules = list(data["all_rules"].values)

    def __getitem__(self, idx):
        if self.add_ling_features:
            input_id, attention_mask, label, rule = self.input_ids[idx], self.attention_masks[idx], self.labels[idx], self.rules[idx]
            return np.array(input_id), np.array(attention_mask), label, self.gru_token, np.array(rule)
        else:
            input_id, attention_mask, label = self.input_ids[idx], self.attention_masks[idx], self.labels[idx]
            return np.array(input_id), np.array(attention_mask), label, self.gru_token, None

    def _get_prediction_results(self, preds):
        df = pd.DataFrame(data={
            "id": self.idxs,
            "prediction": preds,
            "label": self.labels
        })
        return df
