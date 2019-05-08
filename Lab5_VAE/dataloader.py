import random

class dataLoader():
    def __init__(self, root, mode):
        self.root = root
        self.mode = mode

        if self.mode == 'train':
            path = self.root + 'train.txt'
        else:
            path = self.root + 'test.txt'

        fp = open(path, 'r')
        line = fp.readline()
        i = 0
        dict = {}

        while line:
            dict[i] = line.split()
            line = fp.readline()
            i += 1
        print("Total {} vocas.".format(i))
        fp.close()

        self.dict = dict

        self.char2idx = {"GO": 0, "EOS":1}
        self.n_char = 2
        self.idx2char = {0: "GO", 1: "EOS"}

    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.n_char
            self.idx2char[self.n_char] = char
            self.n_char += 1

    def __getitem__(self, index):
        # vector
        pair = random.sample([x for x in range(4)], 2)

        vector_form = [[],[]]

        if self.n_char < 28:
            for i in range(2):
                for c in self.dict[index][pair[i]]:
                    self.add_char(c)
                    vector_form[i].append(self.char2idx[c])
        else:
            for i in range(2):
                for c in self.dict[index][pair[i]]:
                    vector_form[i].append(self.char2idx[c])

        return vector_form

    def index2char(self, vector_form):
        str_form = ['','']
        for i in range(2):
            for num in vector_form[i]:
                str_form[i] = str_form[i] + self.idx2char[num]

        return str_form
                

        
