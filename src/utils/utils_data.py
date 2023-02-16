import random
import torch
from torch.utils.data import Dataset



class DLoader(Dataset):
    def __init__(self, data, tokenizer, config):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = config.max_len
        self.len_per_s = self.max_len // 2

        self.pad_token_id = self.tokenizer.pad_token_id
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.unk_token_id = self.tokenizer.unk_token_id
        self.msk_token_id = self.tokenizer.msk_token_id
        self.special_token_ids = [self.pad_token_id, self.cls_token_id, self.sep_token_id, self.unk_token_id, self.msk_token_id]
        self.vocab_set = set(list(range(self.tokenizer.vocab_size))) - set(self.special_token_ids)

        self.length = len(self.data)


    def random_nsp(self, idx):
        s1, s2 = self.data[idx]
        if random.random() > 0.5:    
            return s1, s2, 1
        return s1, self.get_new_s2(idx), 0

    
    def random_mlm(self, s1, s2):
        # s1 and s2 are sliced from back and front, respectively.
        s1, s2 = [self.cls_token_id] + self.tokenizer.encode(s1)[-self.len_per_s+2:] + [self.sep_token_id], self.tokenizer.encode(s2)[:self.len_per_s-1] + [self.sep_token_id]
        
        # make segments ids
        segment = [1] * len(s1) + [2] * len(s2)

        # do MLM for 15% tokens 
        total_s = s1 + s2
        s_len = len(total_s)
        mlm_len = int((s_len - 3) * 0.15)
        mlm_label = [self.pad_token_id] * s_len
        
        # select MLM position except for special tokens
        mlm_idx = set()
        while len(mlm_idx) != mlm_len:
            tmp_idx = random.randrange(s_len)
            if total_s[tmp_idx] not in self.special_token_ids:
                mlm_idx.add(tmp_idx)

        # do MLM
        for id in list(mlm_idx):
            prob = random.random()

            # change token to [MSK]
            if prob < 0.8:
                mlm_label[id] = total_s[id]
                total_s[id] = self.msk_token_id

            # change token to random token
            elif prob < 0.9:
                new_id = random.choice(list(self.vocab_set - {total_s[id]}))
                mlm_label[id] = total_s[id]
                total_s[id] = new_id

            # remain 10% of tokens
            else:
                mlm_label[id] = total_s[id]
        
        assert len(mlm_label) == len(total_s) == len(segment) == s_len

        # padding
        pad_len = self.max_len - s_len
        total_s = total_s + [self.pad_token_id] * pad_len
        mlm_label = mlm_label + [self.pad_token_id] * pad_len
        segment = segment + [self.pad_token_id] * pad_len
        return total_s, mlm_label, segment


    def get_new_s2(self, idx):
        while 1:
            new_idx = random.randrange(len(self.data))
            if new_idx != idx:
                return self.data[new_idx][1]


    def __getitem__(self, idx):
        s1, s2, nsp_label = self.random_nsp(idx)
        x, mlm_label, segment = self.random_mlm(s1, s2)
        return torch.LongTensor(x), torch.LongTensor(segment), torch.tensor(nsp_label), torch.LongTensor(mlm_label)

    
    def __len__(self):
        return self.length