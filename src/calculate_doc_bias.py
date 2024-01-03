import collections
import numpy as np


class DocBias():
    """
    a class for calculating gender bias in a document
    """
    def __init__(self, wordlist_path):
        genderwords_feml = []
        genderwords_male = []

        for l in open(wordlist_path):
            vals = l.strip().split(',')
            if vals[1]=='f':
                genderwords_feml.append(vals[0])
            elif vals[1]=='m':
                genderwords_male.append(vals[0])

        self.genderwords_feml = set(genderwords_feml)
        self.genderwords_male = set(genderwords_male)

        # print(len(genderwords_feml), len(genderwords_male))

    @staticmethod
    def get_tokens(text):
        return text.lower().split(" ")

    def get_bias(self, tokens):
        text_cnt = collections.Counter(tokens)

        cnt_feml = 0
        cnt_male = 0
        cnt_logfeml = 0
        cnt_logmale = 0
        for word in text_cnt:
            if word in self.genderwords_feml:
                cnt_feml += text_cnt[word]
                cnt_logfeml += np.log(text_cnt[word] + 1)
            elif word in self.genderwords_male:
                cnt_male += text_cnt[word]
                cnt_logmale += np.log(text_cnt[word] + 1)
        text_len = np.sum(list(text_cnt.values()))

        # bias_tc = (float(cnt_feml - cnt_male), float(cnt_feml), float(cnt_male))
        # bias_tf = (np.log(cnt_feml + 1) - np.log(cnt_male + 1), np.log(cnt_feml + 1), np.log(cnt_male + 1))
        bias_bool = (np.sign(cnt_feml) - np.sign(cnt_male), np.sign(cnt_feml), np.sign(cnt_male))

        return bias_bool
