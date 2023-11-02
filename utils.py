import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

# from nltk.corpus import wordnet
import nltk 
nltk.download('wordnet')


random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"].split()

    for i in range(len(text)):
        # Replace with synonyms with a certain probability
        if random.random() < 0.5:  # You can adjust the probability as needed
            text[i] = replace_with_synonyms(text[i])

        # Introduce typos with a certain probability
        if random.random() < 0.3:  # You can adjust the probability as needed
            text[i] = introduce_typos(text[i])

        
    example["text"] = ' '.join(text)

    ##### YOUR CODE ENDS HERE ######

    return example

# Function to replace words in a sentence with their synonyms
def replace_with_synonyms(word):
    synonyms = wordnet.synsets(word)
    if synonyms:
        synonyms = [syn.lemmas()[0].name() for syn in synonyms]
        if synonyms:
            return random.choice(synonyms)
    return word


# Function to introduce typos into a word
def introduce_typos(word):
    typos = {
        'a': ['q', 's', 'z'],
        'b': ['v', 'n', 'g'],
        'c': ['x', 'd', 'f'],
        'd': ['e', 's', 'c'],
        'e': ['w', 'r', 'd'],
        'f': ['r', 'd', 'c'],
        'g': ['f', 'h', 'v'],
        'h': ['g', 'j', 'b'],
        'i': ['u', 'o', 'k'],
        'j': ['h', 'k', 'n'],
        'k': ['j', 'l', 'm'],
        'l': ['k', 'm', 'p'],
        'm': ['n', 'j', 'l'],
        'n': ['m', 'h', 'j'],
        'o': ['p', 'i', 'k'],
        'p': ['o', 'l', 'm'],
        'q': ['a', 'w'],
        'r': ['e', 't', 'f'],
        's': ['a', 'd', 'z'],
        't': ['r', 'y', 'g'],
        'u': ['y', 'i', 'j'],
        'v': ['c', 'b', 'f'],
        'w': ['q', 'e'],
        'x': ['c', 'z'],
        'y': ['t', 'u', 'g'],
        'z': ['x', 'a', 's'],
    }

    word = list(word)
    for i in range(len(word)):
        if random.random() < 0.1:  # You can adjust the probability as needed
            if word[i] in typos:
                typo_options = typos[word[i]]
                word[i] = random.choice(typo_options)
    return ''.join(word)

def custom_transform_train(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"].split()

    for i in range(len(text)):
        # Replace with synonyms with a certain probability
        if random.random() < 0.5:  # You can adjust the probability as needed
            text[i] = replace_with_synonyms(text[i])

        # Introduce typos with a certain probability
        if random.random() > 0.5:  # You can adjust the probability as needed
            text[i] = introduce_typos(text[i])

        
    example["text"] = ' '.join(text)

    ##### YOUR CODE ENDS HERE ######

    return example
