import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        #字典类型，存放每个word的向量值（按位置编码）
        self.word2idx = {}
        #数组类型，存放所有word
        self.idx2word = []

    def add_word(self, word):
        #相同的word只有一个向量值
        if word not in self.word2idx:
            self.idx2word.append(word)
            #按位置编码
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        #os.path.join()函数：连接两个或更多的路径名组件
        #   1.如果各组件名首字母不包含’/’，则函数会自动加上
        #   2.如果有一个组件是一个绝对路径，则在它之前的所有组件均会被舍弃
        #   3.如果最后一个组件为空，则生成的路径以一个’/’分隔符结尾
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        # assert查条件，不符合就终止程序
        """Tokenizes a text file."""
        # assert os.path.exists(path)##################################3服务器运行报错
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                #读出每行，split默认以空格分割
                words = line.split() + ['<eos>']
                #统计该f中的word数
                tokens += len(words)
                #出现一个新的word，就把它添加到dictionary中，并建立对应的从单词到向量的映射关系
                for word in words:
                    self.dictionary.add_word(word)

        #ids是长度为所有word个数的一个向量，这个for为每个word找出dictionary中存放的映射值，设置对应的ids分量
        with open(path, 'r', encoding="utf8") as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1

        return ids
