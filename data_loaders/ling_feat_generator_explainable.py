import re
import json
import pandas as pd

class InfLinguisticRuleGenerator:
    def __init__(self):
        self.general_rules = [        
            r"\s[IRK](?!istan|stan|ya)(lık|lik|luk|lük)\s(.*?)(yap)",
            r"\s[IRK_KIMLIK](?!istan|stan|ya)\s(.*?)(skandalı)",
            r"\s[KIMLIK](?!istan|stan|ya)(ların|lerin|in|ın|un|ün)?\s(.*?)(işgali)",
            r"\s[IRK_KIMLIK](?!istan|stan|ya)(a|e)\s(bak sen)",
            r"\s[IRK](?!istan|stan|ya)(ların|lerin|in|ın|un|ün)?\s(.*?)(işgali)",
            r"\s[IRK](?!istan|stan|ya)\s(.*?)\s[IRK](?!istan|stan|ya)(lığını|liğini|luğunu|lüğünü)\s(.*?)(yap)",
            r"\s[IRK](?!istan|stan|ya)(ın|in|un|ün)\s(.*?)(uşağı|işbirlikçisi|piyonu|kuklası)(.*?)\s[IRK](?!istan|stan|ya)\s",
            r"\s[IRK](?!istan|stan|ya)(ın|in|un|ün)\s(.*?)(hain|vahşi|insanlık dışı|hunharca|kan donduran|şeytani|sinsi|[ADJBEF])\s(teşebbüsleri|planları|oluşumları|[ADJAFTER])",
            r"\s[IRK](?!istan|stan|ya)\s(.*?)(kurşunlarıyla|bombalarıyla|parasıyla)",
            r"\s[IRK](?!istan|stan|ya)(ın|in|un|ün)\s(.*?)(gerçekleştirdiği|yaptığı)(.*?)(katliam|zulüm|soykırım|[ADJAFTER])",
            r"\s[IRK](?!istan|stan|ya)\s(tarafından)\s(gerçekleştirilen|yapılan)(.*?)(saldırılar|katliam|zulüm|soykırım|[ADJAFTER])",
            r"\s[IRK](?!istan|stan|ya)\s(tarafından)\s[IRK](?!istan|stan|ya)(a|e)(yönelik)?(.*?)(saldırılar|katliam|zulüm|soykırım|[ADJAFTER])",
            r"\s[IRK](?!istan|stan|ya)\s(tarafından)(.*?)(saldırıya|katliama|zulme|soykırıma)(.*?)(maruz kal|uğra)",
            r"\s[IRK]\s(destekli)(.*?)\s[IRK](?!istan|stan|ya)\s(darbesi|saldırıları|katliamı|soykırımı)",
            r"\s[IRK](?!istan|stan|ya)\s(.*?)(öldürdü|katletti|etnik temizlik yaptı|kirletti|bastı|şehit etti)",
            r"\s[IRK](?!istan|stan|ya)\s(tarafından)(.*?)(öldürüldü|katledildi|basıldı|şehit edildi)"
        ]
        self.anti_hs_rules = [
            r"""\s[IRK](?!istan|stan|ya)\s(çeteleri|fanatikler|fanatiği|askerleri|milisleri|militanları|yerleşimciler|milliyetçiler|güçleri|isyanı|yaygaracılığı|polisi)""",
            r"(radikal|ırkçı|fanatik)\s[IRK](?!istan|stan|ya)\s",
            r"\s[IRK](?!istan|stan|ya)(cı|ci|çi|cu|cü|çü)\sterör\sörgütü"
        ]
        
        self.special_patterns = json.load(open("ling_feat_files/special_patterns.json", "r"))
        self.hs_specific_verbs = json.load(open("ling_feat_files/hs_specific_verbs.json", "r"))
        self.anti_hs_patterns = json.load(open("ling_feat_files/anti_hs.json", "r"))
        self.adj_bef_keyword = json.load(open("ling_feat_files/adjacent_before_keyword.json", "r"))
        self.adj_after_keyword = json.load(open("ling_feat_files/adjacent_after_keyword.json", "r"))
        
        self.keywords = json.load(open("ling_feat_files/targeted_keywords.json", "r"))
        self.irk = "(" + "|".join(self.keywords["IRK"]) + ")"
        self.kimlik = "(" + "|".join(self.keywords["KIMLIK"]) + ")"
        self.irk_kimlik = "(" + "|".join(self.keywords["IRK"] + self.keywords["KIMLIK"]) + ")"
        
        self.adjbef = "|".join([item for sublist in list(self.adj_bef_keyword.values()) for item in sublist])
        self.adjafter = "|".join([item for sublist in list(self.adj_after_keyword.values()) for item in sublist])
        
        self.turkish_to_latin_map = {ord("ü"): "u", ord("ö"): "o", ord("ı"): "i", ord("ğ"): "g", ord("ş"): "s", ord("ç"): "c"}
        
        self.__prepare_rules()
    
    def __add_keyword_into_rule(self, rules):
        for i in range(len(rules)):
            if "[IRK_KIMLIK]" in rules[i]:
                rules[i] = rules[i].replace("[IRK_KIMLIK]", self.irk_kimlik)
            if "[IRK]" in rules[i]:
                rules[i] = rules[i].replace("[IRK]", self.irk)
            if "[KIMLIK]" in rules[i]:
                rules[i] = rules[i].replace("[KIMLIK]", self.kimlik)
            if "[ADJBEF]" in rules[i]:
                rules[i] = rules[i].replace("[ADJBEF]", self.adjbef)
            if "[ADJAFTER]" in rules[i]:
                rules[i] = rules[i].replace("[ADJAFTER]", self.adjafter)
        return rules 
    
    def __prepare_rules(self):
        self.general_rules = self.__add_keyword_into_rule(self.general_rules)
        self.anti_hs_rules = self.__add_keyword_into_rule(self.anti_hs_rules)

    def __find_spans(self, patterns, text, key_type, degree, return_patterns):
        detected = False
        compiled_regex = re.compile(patterns)
        for r in compiled_regex.finditer(text):
            if return_patterns:
                self.detected_patterns[key_type].append(
                    {
                        "span": r.span(),
                        "match": r.group(),
                        "degree": degree,
                    }
                )
            detected = True
        return detected

    def __detect_special_pattern(self, text, return_patterns=False):
        if return_patterns:
            self.detected_patterns["special"] = []
        pattern_list = []
        for degree, patterns in self.special_patterns.items():
            if len(patterns) != 0:
                patterns = f"\s({'|'.join(patterns)})|({'|'.join(patterns)})\s"
                patterns = r'{}'.format(patterns)
                detected = self.__find_spans(patterns, text, "special", degree, return_patterns)
                if detected:
                    pattern_list.append(1)
                else:
                    pattern_list.append(0)
            else:
                pattern_list.append(0)
        return pattern_list
    
    def __detect_general_rules(self, text):
        nof_rules_detected = 0
        for rule in self.general_rules:
            found_texts = re.findall(rule, text, flags=re.I)
            if len(found_texts) > 0:
                nof_rules_detected += 1
        return nof_rules_detected / len(self.general_rules)
    
    def __detect_anti_hs(self, text, return_patterns):
        if return_patterns:
            self.detected_patterns["anti_hs"] = []
        pattern_list = []
        for degree, patterns in self.anti_hs_patterns.items():
            if len(patterns) != 0:
                found_rule = False
                if degree == "20":
                    for rule in self.anti_hs_rules:
                        detected = self.__find_spans(rule, text, "anti_hs", degree, return_patterns)
                        if detected:
                            found_rule = True
                else:
                    patterns = f"\s({'|'.join(patterns)})|({'|'.join(patterns)})\s"
                    patterns = r'{}'.format(patterns)
                    detected = self.__find_spans(patterns, text, "anti_hs", degree, return_patterns)
                if detected or found_rule:
                    pattern_list.append(1)
                else:
                    pattern_list.append(0)
            else:
                pattern_list.append(0)
        return pattern_list
        
    def __detect_bef_adj(self, text, return_patterns=False):
        if return_patterns:
            self.detected_patterns["bef_target"] = []
        pattern_list = []
        for degree, patterns in self.adj_bef_keyword.items():
            if len(patterns) != 0:
                patterns = f"\s({'|'.join(patterns)})\s{self.irk}(?!istan|stan|ya)\s"
                patterns = r'{}'.format(patterns)
                detected = self.__find_spans(patterns, text, "bef_target", degree, return_patterns)
                if detected:
                    pattern_list.append(1)
                else:
                    pattern_list.append(0)
            else:
                pattern_list.append(0)
        return pattern_list

    def __detect_after_adj(self, text, return_patterns=False):
        if return_patterns:
            self.detected_patterns["after_target"] = []
        pattern_list = []
        for degree, patterns in self.adj_after_keyword.items():
            if len(patterns) != 0:
                patterns = f"\s{self.irk}(?!istan|stan|ya)\s({'|'.join(patterns)})\s"
                patterns = r'{}'.format(patterns)
                detected = self.__find_spans(patterns, text, "after_target", degree, return_patterns)
                if detected:
                    pattern_list.append(1)
                else:
                    pattern_list.append(0)
            else:
                pattern_list.append(0)
        return pattern_list
    
    def __detect_hs_specific_verbs(self, text, return_patterns=False):
        if return_patterns:
            self.detected_patterns["hs_verbs"] = []
        pattern_list = []
        for degree, patterns in self.hs_specific_verbs.items():
            if len(patterns) != 0:
                found_pattern = False
                patterns = f"\s{self.irk}(?!istan|stan|ya)(.*?)({'|'.join(patterns)})"
                patterns = r'{}'.format(patterns)
                found_texts = re.findall(patterns, text, flags=re.I)
                for found_text in found_texts:
                    if len(found_text[1].split()) > 13:
                        new_text = " ".join(found_text[1].split()[-13:]) + " " + found_text[2]
                        if len(re.findall(patterns, new_text, flags=re.I)) > 0:
                            _ = self.__find_spans(patterns, new_text, "hs_verbs", degree, return_patterns)
                            found_pattern = True
                    else:
                        _ = self.__find_spans(patterns, text, "hs_verbs", degree, return_patterns)
                        found_pattern = True
                        break
                if found_pattern:
                    pattern_list.append(1)
                else:
                    pattern_list.append(0)
            else:
                pattern_list.append(0)
        return pattern_list
        
    def apply_rules(self, data, return_patterns=False):
        self.detected_patterns = {}
        data["special_pattern"] = data["text"].apply(lambda text: self.__detect_special_pattern(text, return_patterns=return_patterns))
        data["general_rule"] = data["text"].apply(lambda text: self.__detect_general_rules(text))
        data["anti_hs"] = data["text"].apply(lambda text: self.__detect_anti_hs(text, return_patterns=return_patterns))
        data["hs_specific_verb"] = data["text"].apply(lambda text: self.__detect_hs_specific_verbs(text, return_patterns=return_patterns))
        data["adj_bef_keyword"] = data["text"].apply(lambda text: self.__detect_bef_adj(text, return_patterns=return_patterns))
        data["adj_after_keyword"] = data["text"].apply(lambda text: self.__detect_after_adj(text, return_patterns=return_patterns))
        if return_patterns:
            return data, self.detected_patterns
        else:
            return data

if __name__ == '__main__':
    data_path = ""
    data = pd.read_csv(data_path, sep='|')
    rule_assigner = InfLinguisticRuleGenerator()
    data, detected_patterns =  rule_assigner.apply_rules(data, return_patterns=True)
    print(data.head())
    print(detected_patterns)
