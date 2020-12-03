# Load libraries
import os
import unicodedata
import re
import itertools
import pandas as pd
import numpy as np
import gensim
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
import pickle
import argparse

# Load functions:
symbols_to_space = re.compile(u"[/\|\n(\; )|(\: )|( \()|(\) )|( \")|(\" )|( \')|(\' )|\t]")
symbols_to_remove = re.compile(u"[\"\'\$\€\£\(\)\:\[\]\.\,\>\<\?\-\_]")
space_repetition = re.compile(u" {2,}")
words_with_numbers = re.compile(u"\w*\d\w*")


def canonize_language(text):
    """
    Function to remove spaces, symbols...
    """
    text = strip_accents(text.strip().lower())
    text = symbols_to_space.sub(" ", text)
    text = symbols_to_remove.sub("", text)
    text = space_repetition.sub(" ", text)
    text = words_with_numbers.sub("", text)
    return text.strip()


def simple_tokenizer(text, min_token_length=0):
    """
    Function to short words
    """
    tokens = text.split(" ")
    if min_token_length > 0:
        tokens = filter(lambda x: len(x) >= min_token_length, tokens)
    return tokens


def strip_accents(input_str):
    """
    Remove accents
    """
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return nfkd_form


def lem_word(txt_list, lemmer):
    """
    Lemmatizer of words
    """
    return lemmer.lemmatize(txt_list, "v")


def join_lines_txt(txt_list, separator=' '):
    """
    Join element words of a list into one string
    """
    return separator.join(txt_list)


def annotation_weight_representation(venue_annotation):
    """
    Function that iterates a long a list vocab. Look up for its embedding. Make weighted avg of embeddings 
    """
    venue_vectors = []
    count_model_included = 0
    count_model_nonincluded = 0
    idx_token_vectors = 0

    tags = venue_annotation.split()
    word_vectors = np.empty(shape=(len(tags), 300))
    idx_word_vectors = 0
    for tag in tags:
        tag = tag.replace('_', ' ')
        if tag in model.vocab:
            word_vectors[idx_word_vectors] = model[tag]
            idx_word_vectors += 1
        else:
            tokens = tag.split()
            token_vectors = np.empty(shape=(len(tokens), 300))
            idx_token_vectors = 0
            for token in tokens:
                if token in model.vocab:
                    token_vectors[idx_token_vectors] = model[token]
                    idx_token_vectors += 1
                else:
                    continue
            if idx_token_vectors > 0:
                word_vectors[idx_word_vectors] = np.average(token_vectors[:idx_token_vectors], axis=0)
                idx_word_vectors += 1

    if idx_word_vectors != 0 or idx_token_vectors != 0:
        count_model_included += 1
        venue_vectors.append(np.average(word_vectors[:idx_word_vectors], axis=0))
    else:
        count_model_nonincluded += 1
        venue_vectors.append(np.nan)

    return venue_vectors[0]


def model_load(name):
    f = open(str(name)+'.pkl', 'rb')
    model = pickle.load(f)
    f.close()
    return model



def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Predict')

    parser.add_argument(
        '--model',
        dest='classifier',
        default='classifier',
        required=True)

    parser.add_argument(
        'files',
        metavar='files',
        type=str,
        nargs='+',
        help='process data from each file')

    return parser.parse_args()



if __name__ == '__main__':

    args = parse_args()
    initial_path = os.getcwd()

    #Load model:
    classifier=model_load(args.classifier)
    model=gensim.models.KeyedVectors.load_word2vec_format(os.path.join(initial_path,"GoogleNews-vectors-negative300.bin") ,binary=True)
	
    #Set fix parameters
    n_chars = 4
    stemmer = PorterStemmer()
    lemmer = WordNetLemmatizer()
    other_words = ["@", "that", "than", "=", "*", "^", "+", "-", "these"]
    		
    #Print each file prediction
    for i_file in args.files:
        clean_txt=[]
        with open(i_file) as file:
            text_data = file.readlines()
        for i_sub_txt in np.arange(0, len(text_data)):
            i_line = text_data[i_sub_txt]
            txt_f1 = canonize_language(i_line)
            txt_f2 = simple_tokenizer(txt_f1, n_chars)
            txt_f2 = list(txt_f2)
            lemmed_vector = [lem_word(element_txt, lemmer) for element_txt in txt_f2] # look up at wordnet dictionary
            lemmed_vector = [ele for ele in lemmed_vector if all(ch not in ele for ch in other_words)] #Delete words containing @ (Dont know if its good idea since i dont know if @mails correlate with the categories will be uncommented if in the analysis is not a clue)
            lemmed_vector=join_lines_txt(lemmed_vector)
            clean_txt.append(lemmed_vector)
        clean_merged_txt = join_lines_txt(clean_txt)
        clean_merged_txt = clean_merged_txt.strip()
        clean_merged_txt = space_repetition.sub(" ", clean_merged_txt)
        transmissions_df = pd.DataFrame([clean_merged_txt], columns=["transmissions"])
        transmissions_df["vector_rep"] = transmissions_df["transmissions"].apply(lambda x: annotation_weight_representation(x))
        embeddings_df = transmissions_df['vector_rep'].apply(pd.Series)
        embeddings_df = embeddings_df.rename(columns = lambda x : 'element_' + str(x))
        print(os.path.basename(os.path.normpath(i_file)), classifier.predict(embeddings_df)[0])