import re
from typing import List
import os

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
    """split the sentence into N-gram

    Args:
        sentence (str): the sentence input 
        N (int, optional): the n parameter. Defaults to 2
        
    Returns:    
        list(str): list of substrings
    """
    return [sentence[i:i+N] for i in range(len(sentence)-N+1)]

def padding_list(someList: List[str], N: int) -> List[str]:
    """Padding the list with <s> at the front and </s> behind

    Args:
        someList (List[str]): The list to be padded with
        N (int): The amount of <s>, </s> to be padded

    Returns:
        List[str]: Padded list
    """
    for i in range(N):
        someList = ['<s>'] + someList + ['</s>']
    return someList

# Moving files into the testing dataset
def move_files(from_dir: str, to_dir: str, N: int):
    """Move N text files from from_dir to to_dir

    Args:
        from_dir (str): the current directory
        to_dir (str): the target directory
        N (int): amount of files sent
    """
    directory = os.fsencode(from_dir)
    for index, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            from_file = str(os.path.join(from_dir, filename))
            to_file = str(os.path.join(to_dir, filename))
            os.rename(from_file, to_file) 
        if index == N-1:
            break