import torch
from torch.utils.data import Dataset

class Sentences(Dataset):
    def __init__(self, load_path='./data/sentences.txt'):
        f = open(load_path)
        sentences = f.read().split('\n')
        self.sentences = []
        for sentence in sentences:
            sentence = sentence.lower()
            self.sentences.append(sentence)
        f.close()
        self.num = len(self.sentences)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        return self.sentences[idx]

    def random_sentence(self):
        return self.sentences[torch.randint(self.num, (1,))]

if __name__ == '__main__':
    sentences = Sentences()
    print(len(sentences))
    x = sentences[3]
    print(x)

    print(sentences.random_sentence())