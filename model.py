from collections import Counter, namedtuple
import json
import re

def ngram(documents: list, N: int=2) -> dict:
    """Build a n-gram probability dictionary from the document list

    Args:
        documents (list): the traning documents that are
            placed in a list.
        N (int, optional): the n-gram parameter.
            Defaults to 2.

    Returns:
        dict: [description]
    """
    ngram_prediction = dict()
    total_grams = list()
    words = list()
    Word = namedtuple('Word', ['word', 'prob'])

    for doc in documents:
        split_words = ['<s>'] + list(doc) + ['</s>']
        # 計算分子
        [total_grams.append(tuple(split_words[i:i+N])) for i in range(len(split_words)-N+1)]
        # 計算分母
        [words.append(tuple(split_words[i:i+N-1])) for i in range(len(split_words)-N+2)]
        
    total_word_counter = Counter(total_grams)
    word_counter = Counter(words)
    
    for key in total_word_counter:
        word = ''.join(key[:N-1])
        if word not in ngram_prediction:
            ngram_prediction.update({word: set()})
            
        next_word_prob = total_word_counter[key]/word_counter[key[:N-1]]
        w = Word(key[-1], '{:.3g}'.format(next_word_prob))
        ngram_prediction[word].add(w)
        
    return ngram_prediction

if __name__ == "__main__":
    # get the datas for training
    DATASET_DIR = './WebNews.json'
    with open(DATASET_DIR, encoding = 'utf8') as f:
        dataset = json.load(f)
    
    seg_list = list(map(lambda d: d['detailcontent'], dataset))
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    seg_list = [rule.sub('', seg) for seg in seg_list]

    tri_prediction = ngram(seg_list, N=3)
    for word, ng in tri_prediction.items():
        tri_prediction[word] = sorted(ng, key=lambda x: x.prob, reverse=True)

    text = '韓國'
    next_words = list(tri_prediction[text])
    for next_word in next_words:
        print('next word: {}, probability: {}'.format(next_word.word, next_word.prob))