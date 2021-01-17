from opencc import OpenCC
'''pip install opencc-python-reimplemented'''

simplified_to_traditional = lambda text : OpenCC('s2t').convert(text)