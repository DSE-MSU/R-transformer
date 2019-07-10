import unidecode
import torch
from collections import Counter
import observations
import os
import pickle

cuda = torch.cuda.is_available()


def data_generator(args, data_dir):
    file, testfile, valfile = getattr(observations, args.dataset)(data_dir)
    file_len = len(file)
    valfile_len = len(valfile)
    testfile_len = len(testfile)
    corpus = Corpus(file + " " + valfile + " " + testfile)

    return file, file_len, valfile, valfile_len, testfile, testfile_len, corpus


def read_file(filename):
    file = unidecode.unidecode(open(filename).read())
    return file, len(file)


class Dictionary(object):
    def __init__(self):
        self.char2idx = {}
        self.idx2char = []
        self.counter = Counter()

    def add_word(self, char):
        self.counter[char] += 1

    def prep_dict(self):
        for char in self.counter:
            if char not in self.char2idx:
                self.idx2char.append(char)
                self.char2idx[char] = len(self.idx2char) - 1

    def __len__(self):
        return len(self.idx2char)


class Corpus(object):
    def __init__(self, string):
        self.dict = Dictionary()
        for c in string:
            self.dict.add_word(c)
        self.dict.prep_dict()


def char_tensor(corpus, string):
    tensor = torch.zeros(len(string)).long()
    for i in range(len(string)):
        tensor[i] = corpus.dict.char2idx[string[i]]
    return tensor.cuda()


def batchify(data, batch_size):
    """The output should have size [L x batch_size], where L could be a long sequence length"""
    # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
    nbatch = data.size(0) // batch_size
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * batch_size)
    # Evenly divide the data across the batch_size batches.
    data = data.view(batch_size, -1)
    return data


def get_batch(source, start_index, args):
    seq_len = min(args.seq_len, source.size(1) - 1 - start_index)
    end_index = start_index + seq_len
    inp = source[:, start_index:end_index].contiguous()
    target = source[:, start_index+1:end_index+1].contiguous()  # The successors of the inp.
    return inp, target


def save(model, save_filename):
    torch.save(model, save_filename)
    print('Saved as %s' % save_filename)


def output_s(message, save_filename):
    with open(save_filename, 'a') as out:
        out.write(message + '\n')
    


