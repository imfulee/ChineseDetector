import re
from nltk import ngrams
from typing import List, Set, Dict, Tuple, Optional

def clean_string(sentence: str) -> str:
    """Clear the string of non-chinese characters

    Args:
        sentence (str): the sentence input

    Returns:
        str: cleaned non-chinese characters
    """
    rule = re.compile(r"[^\u4e00-\u9fa5]")
    clean_str = rule.sub('', sentence)
    return clean_str

def build_ngram(sentence: str, N: int = 2) -> List[str]:
    """split the sentence into N -gram

    Args:
        sentence (str): the sentence input 
        N (int, optional): the n parameter. Defaults to 2
        
    Returns:    
        list(str): list of substrings
    """
    return [sentence[i:i+N] for i in range(len(sentence)-N+1)]
