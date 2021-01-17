from collections import Counter
from preprocessor import clean_string, build_ngram
import math

class Ngram():
    def __init__(self, N: int=2):
        self.N = N
        self.ngram_prediction = dict()
        self.N_given = 0
        self.N_all = 0
        self.B_given = 0
        self.B_all = 0
        
    def fit_transform(self, documents: list) -> dict:
        """Build a n-gram probability dictionary from the document list
        Args:
            documents (list): the traning documents that are
                placed in a list.
            N (int, optional): the n-gram parameter.
                Defaults to 2.
        Returns:
            dict: the dictionary with all the n-grams.
        """
        N = self.N
        
        total_grams = list()
        total_unigrams = list()
        words = list()
        #Word = namedtuple('Word', ['word', 'numerator','denominator'])


        for doc in documents:
            split_words = ['<s>'] + list(doc) + ['</s>'] # TODO:possible bug for trigram
            # 計算分子
            [total_grams.append(tuple(split_words[i:i+N])) for i in range(len(split_words)-N+1)]
            # 計算分母
            [words.append(tuple(split_words[i:i+N-1])) for i in range(len(split_words)-N+2)]
            # 計算unigrams 數量
            [total_unigrams.append(tuple(split_words[i])) for i in range(len(split_words))]

        # self.N_given: count of all N-1 gram
        # self.N_all:   count of all N gram
        # self.B_given: count of possible N-1 gram
        # self.B_all:   count of possible N gram
        self.N_given = len(total_grams)
        self.N_all = len(words)
        N_uni = len(list(set(total_unigrams)))
        self.B_given = int(math.pow(N_uni, N-1))
        self.B_all = int(math.pow(N_uni, N))

        total_word_counter = Counter(total_grams)
        word_counter = Counter(words)
        for key in total_word_counter:
            word = ''.join(key[:N-1])
            if word not in self.ngram_prediction:
                self.ngram_prediction.update({word: dict()})
                self.ngram_prediction[word]['count'] = word_counter[key[:N-1]]

            numerator = total_word_counter[key]  # ---------------term 2 接著 term 1 出現的次數
            self.ngram_prediction[word][key[-1]] = numerator
            
        #print(self.N_given, self.N_all, N_uni, self.B_given, self.B_all)

        return self.ngram_prediction


    def string_prob(self, sentence: str, gamma: float=0.5, verbose: bool=False) -> float:
        """Calculate the probability of the string happening for
        prediction n-gram dictionary
        Args:
            sentence (str): the sentence input
            prediction (dict): the n-gram prediction dictionary
        Returns:
            float: the probability of the string happening for
                prediction n-gram dictionary
        """
        N = self.N
        prediction = self.ngram_prediction

        clean_str = clean_string(sentence)
        gram_str = build_ngram(clean_str, N) # n-gram of the prediction string
        predicting_str = list(clean_str) # list used for prediction
        probability = 1 # the probability of the sentence from n-gram dict

        # N-gram Lidstone (add one smoothing)
        # P(q|d) = mulplipy_all( P(term|d) for term(N-gram) in q )

        for text in gram_str:
            prefix = text[:-1]
            last_word = text[-1]
            if(verbose):
                print("prefix is ({:s}) and last_word is ({:s})".format(prefix, last_word))
            if(prefix not in prediction.keys()):
                P_all = 1/self.B_all
                P_given = 1/self.B_given
            elif(last_word not in prediction[prefix].keys()):
                P_all = gamma/(self.N_all + self.B_all*gamma)
                P_given = gamma/(self.N_given + self.B_given*gamma)
            else:
                P_all = (prediction[prefix][last_word] + gamma)/(self.N_all + self.B_all*gamma)
                P_given = (prediction[prefix]['count'] + gamma)/(self.N_given + self.B_given*gamma)
            if(verbose):
                print("all count =", P_all * (self.N_all + self.B_all*gamma))
                print("given count =", P_given * (self.N_given + self.B_given*gamma))
                print("all prob = {:.3f}".format(P_all))
                print("given prob = {:.3f}".format(P_given))
            probability *= P_all/P_given
            
        return probability


if __name__ == "__main__": 

    def files_to_list(folder_directory: str) -> list:
        """Read the files in the folder with .txt to a list

        Args:
            directory (str): the directory to find files

        Returns:
            list: list with all the files turned into str
        """
        dataset = []
        
        import os

        directory = os.fsencode(folder_directory)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".txt"): 
                with open(str(os.path.join(folder_directory, filename)), 'r') as doc:
                    document_txt = doc.read()
                    dataset.append(document_txt)

        return [clean_string(doc) for doc in dataset]    

    print("Process: Read the files")
    china_dataset = files_to_list('./ChinaDataset/')
    taiwan_dataset = files_to_list('./TaiwanDataset/')
    
    n_parameter = 2
    
    print("Process: Train the models")
    # Train the China n-gram model
    china_ngram = Ngram(N=n_parameter)
    china_predict = china_ngram.fit_transform(china_dataset)
    
    # Train the Taiwan n-gram model
    taiwan_ngram = Ngram(N=n_parameter)
    taiwan_predict = taiwan_ngram.fit_transform(taiwan_dataset)
    
    sentence = ''
    verbose_mode = False
    # Classifier
    while True:    
        sentence = input("Sentence: ")
        if sentence == 'quit':
            break
        china_prob = china_ngram.string_prob(sentence, verbose=verbose_mode)
        taiwan_prob = taiwan_ngram.string_prob(sentence, verbose=verbose_mode)
        ratio = taiwan_prob / china_prob
        print(f'Ratio: {ratio}', end=' -> ')
        if ratio > 1:
            print("Taiwan", end='\n\n')
        elif ratio < 1:
            print("China", end='\n\n')
        else:
            print("Undecided", end='\n\n')