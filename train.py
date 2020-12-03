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


def transmission_categories_files(main_path):
    """
    Function to read category folders
    """
    cats = []
    try:
        cats = os.listdir(main_path)
        num_cats = len(cats)
        print("---There are {} categories to train---".format(num_cats)) if num_cats > 0 else print(
            "---There is not any category folder, add some please!")
    except ValueError:
        print("---It does not exist that path, create it or change it!")
    return cats


def files_corrector(main_path):
    """
    Function to change rare name folders
    """
    print("---Formatting file names---")
    for i_cat in cats:
        cat_path = os.path.join(main_path, i_cat)
        os.chdir(cat_path)
        txt_cat_files = os.listdir(cat_path)
        for i_file in txt_cat_files:
            if bool(re.search("^.!", i_file)):
                prev_name = re.search('^.![^!]+!', i_file).group(0)
                new_name = re.sub('\.|\!', '', prev_name)
                os.rename(i_file, new_name)
    os.chdir(main_path)
    print("DONE")
    pass


def text_to_dict(main_path, cats):
    """
    Function to put txt into a dictionary
    """
    pruebas_dict = []
    try:
        for i_cat in cats:
            print("---Loading cat {} ---".format(i_cat))
            cat_path = os.path.join(main_path, i_cat)
            os.chdir(cat_path)
            txt_cat_files = os.listdir(cat_path)
            for i_file in txt_cat_files:
                with open(i_file) as file:
                    text_data = file.readlines()
                    aux_text_dict = {i_cat: text_data}  # Podremos acceder a ellos a través de .keys o .values
                pruebas_dict.append(aux_text_dict)
    except ValueError:
        print("---Still with no category folders in that path. Please to continue add the data or change the path")
    os.chdir(main_path)
    return pruebas_dict


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


# lemmer = WordNetLemmatizer()

def lem_word(txt_list, lemmer):
    """
    Lemmatizer of words
    """
    return lemmer.lemmatize(txt_list, "v")


# stemmer = PorterStemmer()
def stem_word(txt_list, stemmer):
    """
    Stemmer for words
    """
    return stemmer.stem(txt_list)


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


def parse_args():
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train Classifier')

    parser.add_argument(
        '--dataset',
        dest='dataset_folder',
        default='dataset',
        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # Read data
    name_folder = args.dataset_folder
    initial_path = os.getcwd()
    main_path = os.path.join(os.getcwd(), name_folder)
    cats = transmission_categories_files(main_path)
    files_corrector(main_path)
    data_files = text_to_dict(main_path, cats)
    # Clean data
    clean_data = []
    n_chars = 4
    stemmer = PorterStemmer()
    lemmer = WordNetLemmatizer()
    other_words = ["@", "that", "than", "=", "*", "^", "+", "-", "these"]
    # Iterate through each doc
    for i_txt in np.arange(0, len(data_files)):
        txt = list(data_files[i_txt].values())[0]
        clean_txt = []
	    # Iterate lines of each doc and join them in one
        for i_sub_txt in np.arange(1, len(txt)):
            i_line = txt[i_sub_txt]
            txt_f1 = canonize_language(i_line)
            txt_f2 = simple_tokenizer(txt_f1, n_chars)
            txt_f2 = list(txt_f2)
            lemmed_vector = [lem_word(element_txt, lemmer) for element_txt in txt_f2]  # look up at wordnet dictionary
            lemmed_vector = [ele for ele in lemmed_vector if all(ch not in ele for ch in
                                                                 other_words)]  # Delete words containing @ (Dont know if its good idea since i dont know if @mails correlate with the categories will be uncommented if in the analysis is not a clue)
            lemmed_vector = join_lines_txt(lemmed_vector)
            clean_txt.append(lemmed_vector)
        clean_merged_txt = join_lines_txt(clean_txt)
        # SECOND CLEANING STEP after merging:
        # ----------------------
        clean_merged_txt = clean_merged_txt.strip()
        clean_merged_txt = space_repetition.sub(" ", clean_merged_txt)
        aux_clean_data = {list(data_files[i_txt].keys())[0]: clean_merged_txt}
        clean_data.append(aux_clean_data)
    # Tabular data
    transmissions_df = pd.DataFrame()
    for irow in np.arange(1, len(clean_data)):
        auxrow = clean_data[irow]
        target = list(auxrow.keys())[0]
        transmissions = list(auxrow.values())[0]
        aux_pd = pd.DataFrame([[transmissions, target]], columns=["transmissions", "target"])
        transmissions_df = transmissions_df.append(aux_pd)
    transmissions_df.reset_index(inplace=True)
    transmissions_df.drop(columns="index", inplace=True)
    # Prepare data:
    transmissions_df = transmissions_df.reindex(np.random.permutation(transmissions_df.index)).reset_index(drop=True)
    model = gensim.models.KeyedVectors.load_word2vec_format(
        os.path.join(initial_path, "GoogleNews-vectors-negative300.bin"), binary=True)
    transmissions_df["vector_rep"] = transmissions_df["transmissions"].apply(
        lambda x: annotation_weight_representation(x))
    # Expand the elements embedding as variables and stick the target variable to have our master table:
    embeddings_df = transmissions_df['vector_rep'].apply(pd.Series)
    embeddings_df = embeddings_df.rename(columns=lambda x: 'element_' + str(x))
    embeddings_df = pd.merge(embeddings_df, transmissions_df["target"], left_index=True, right_index=True)
    # Split in train/test
    sz = embeddings_df.shape
    train = embeddings_df.iloc[:int(sz[0] * 0.8), :]
    test = embeddings_df.iloc[int(sz[0] * 0.8):, :]
    train = train.dropna()
    test = test.dropna()
    # Save the train and test data sets with its label
    x_train = train.iloc[:, :300]
    y_train = train.iloc[:, 300]
    x_test = test.iloc[:, :300]
    y_test = test.iloc[:, 300]
    rf_clf = RandomForestClassifier(n_estimators=1500, class_weight="balanced", max_depth=5, random_state=655321)
    model = rf_clf.fit(x_train, y_train)
    # Save model
    output = open(os.path.join(initial_path, 'classifier.pkl'), 'wb')
    pickle.dump(model, output)
    output.close()
