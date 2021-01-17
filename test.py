from preprocessor import *
from ngram import *
sentence = "是韓國人"
sentence2 = "好韓國人"
docs = [
    "我是韓國人",
    "我不是中國人",
    "你好臭我是ㄐㄐ"
    ]
print("search: ",sentence)
print("docs: ", docs)


ngram = Ngram(N = 2)
predict = ngram.fit_transform(docs)
for word in predict:
    print(word, predict[word])


prob = ngram.string_prob(sentence, verbose=True)
print()
prob2 = ngram.string_prob(sentence2, verbose=True)
print("prob1:",prob)
print("prob2:",prob2)