# Chinese Detector

A text mining model that uses `N-gram` models (in this instance 3) to detects if someone is from Taiwan or China. Originally called `共匪測試機`, and changed as the name was not being very friendly to our overseas neighbours and potential overlords.

## Goals

There are two overall goals for this project:

1. Calculate the probability of whether a sequence of strings is more likely to be from Taiwan or China
2. Predict the next character / string from a given string

## Set up 

Install `python 3`, `pip` and (optionally) `venv` on your computer and the required packages from the `requirements.txt` file. As of writing this, there is no need to install anything other than `python 3`.

## Test environment

This is tested on `python 3.8.6` on `Ubuntu 20.10` and `Windows` but most likely there would not be any problems if you're using `python 3.x` or running `Mac OS`. 

## Implementation

As of now, we do not use the `nltk` package for our purposes, rather we wrote out own implementation of 

- tokenization
- building the n-gram 
- smoothing technique (Lidstone's Law)
- some kind of classifier

The is a wish we could pivot to use more standard packages in the future.

## Usage

To use this, prepare a bunch of documents (`.txt` files) that are from China and Taiwan and seperate them in two folders (The default preset is `ChinaDataset/` and `TaiwanDataset/`). You could change the folder directories in `ngram.py`. 

```python
# change the directories if you wish
china_dataset = files_to_list('./ChinaDataset/')
taiwan_dataset = files_to_list('./TaiwanDataset/')
```

Then just use your terminal/command line and type 

```bash
python3 ngram.py
```

and type in the sentence you wish to check

```bash
Sentence: 我是從火星來的
```

## Credits

Parts of my code comes from the articles I have read online and I may miss out on some credits. So, if you see your code used and not credited here, please do tell.

- [A Comprehensive Guide to Build your own Language Model in Python! - Mohd Sanad Zaki Rizvi](https://medium.com/analytics-vidhya/a-comprehensive-guide-to-build-your-own-language-model-in-python-5141b3917d6d)
- [Building Language Models for Text with Named Entities - Md Rizwan Parvez, Saikat Chakraborty, Baishakhi Ray, Kai-Wei Chang](https://arxiv.org/abs/1805.04836)
- [Building language models - bogdani](https://nlpforhackers.io/language-models/)
- [自然語言處理 — 使用 N-gram 實現輸入文字預測 - Airwaves](https://medium.com/%E6%89%8B%E5%AF%AB%E7%AD%86%E8%A8%98/%E8%87%AA%E7%84%B6%E8%AA%9E%E8%A8%80%E8%99%95%E7%90%86-%E4%BD%BF%E7%94%A8-n-gram-%E5%AF%A6%E7%8F%BE%E8%BC%B8%E5%85%A5%E6%96%87%E5%AD%97%E9%A0%90%E6%B8%AC-10ac622aab7a)
- [结巴:中文分词组件](https://github.com/fxsjy/jieba)

## TODO

- add `jieba` for (possibly) better tokenization