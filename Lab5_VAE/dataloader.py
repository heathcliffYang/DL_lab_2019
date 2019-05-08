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
        fp.close()

        self.dict = dict

        self.char2idx = {"GO": 0, "EOS":1}
        self.n_char = 2
        self.idx2char = {0: "GO", 1: "EOS"}
        self.total_words = i
        # Go over
        if self.mode == 'train':
            print("Go over", i, "words.")
            for j in range(i):
                self[j]
            print("End.")
            

    def def_char_i_dict(self, char2idx, idx2char, n_char):
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.n_char = n_char
        print("Go over", self.total_words, "words.")
        for j in range(self.total_words):
            self[j]
        print("End.")

    def add_char(self, char):
        if char not in self.char2idx:
            self.char2idx[char] = self.n_char
            self.idx2char[self.n_char] = char
            self.n_char += 1

    def __getitem__(self, index):
        # vector
        # pair = random.choice([
        #         [0, 3],
        #         [0, 2],
        #         [0, 1],
        #         [0, 1],
        #         [3, 1],
        #         [0, 2],
        #         [3, 0],
        #         [2, 0],
        #         [2, 3],
        #         [2, 1],
        #     ])
        # print(pair)

        if self.mode == 'train':
            i = random.randint(0, 3)
            pair = [i,i]
        elif self.mode == 'ge':
            pair = random.sample([x for x in range(4)], 2)
        else:
            pair = [0, 1]

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

        if self.mode == 'test':
            pair_set = [
                [0, 3],
                [0, 2],
                [0, 1],
                [0, 1],
                [3, 1],
                [0, 2],
                [3, 0],
                [2, 0],
                [2, 3],
                [2, 1],
            ]
            pair = pair_set[index]

        return vector_form, pair


    def vector2char(self, vector_form):
        str_form = ''
        for num in vector_form:
            str_form = str_form + self.idx2char[num]

        return str_form

    def __len__(self):
        return self.total_words