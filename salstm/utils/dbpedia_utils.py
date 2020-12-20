"""
   Code adapted from
   https://github.com/dongjun-Lee/transfer-learning-text-tf/blob/master/data_utils.py
   Utilities to handle the Dbpedia dataset.
"""
import collections
import os
import re
import string
import tarfile
from functools import partial

import numpy as np

import nltk
import pandas as pd
import wget
from nltk.corpus import words
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from salstm.models.tokens import EOS_TOKEN, PAD_TOKEN, SOS_TOKEN, UNK_TOKEN
from salstm.utils import file_utils


def download_dbpedia():
    dbpedia_url = 'https://github.com/le-scientifique/torchDatasets/raw/master/dbpedia_csv.tar.gz'

    wget.download(dbpedia_url)
    with tarfile.open("dbpedia_csv.tar.gz", "r:gz") as tar:
        tar.extractall()


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?:;\-\'\`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip().lower()

    return text


def english_words():
    try:
        eng_words = set(words.words())
    except LookupError:
        print("LookupError when trying to access <nltk.words.words()> function")
        nltk.download('words')
        eng_words = set(words.words())

    return eng_words


def reverse_word_dictionary(word_dict):
    return {id_: token for token, id_ in word_dict.items()}


def build_word_dictionary_with_dict(input_dict, output_file):

    if not os.path.exists(output_file):
        word_dict = dict()
        word_dict[PAD_TOKEN] = 0
        word_dict[UNK_TOKEN] = 1
        word_dict[SOS_TOKEN] = 2
        word_dict[EOS_TOKEN] = 3

        print("{} keys in <required_dict> parameter".format(len(input_dict)))
        added_words = 0
        for current_key in input_dict:
            if current_key not in word_dict:
                word_dict[current_key] = len(word_dict)
                added_words += 1
        print("{} words added to word_dict".format(added_words))

        file_utils.save(filename=output_file, data=word_dict)
    else:
        print("Loading word_dict from <{}>".format(output_file))
        word_dict = file_utils.load(filename=output_file)

    return word_dict


def build_word_dictionary(path_to_data, output_file, min_frequency=1, required_dict=None):
    """

    :param path_to_data: file containing the dbpedia data
    :param output_file: file where we are going to save the word dictionary produced
    :param min_frequency: minimum frequency of a word to be considered to
       inclusion in the word dictionary
    :param required_dict: dictionary {key: value} or {word: id_number}
    :return:
    """
    if not os.path.exists(output_file):
        train_df = pd.read_csv(path_to_data, names=["class", "title", "content"])
        contents = train_df["content"]

        # Using only word_tokenize generates an unwanted behaviour:
        # words are still not tokenized correctly.
        # Example: la la la 1884. bla. -> la, la, la, 1884., bla, .
        # Solution: word_tokenize from
        # https://github.com/nltk/nltk/issues/1963#issuecomment-367185040
        # Alternative: nltk.wordpunct_tokenize(sent)
        # punct_tokenize = partial(re.sub, pattern='([.,!?(){}]+)', repl=' \g<1> ', flags=re.U)
        current_words = list()
        for content in contents:
            # word_tokenize(punct_tokenize(string=clean_str(content))):
            for word in wordpunct_tokenize(clean_str(content)):
                current_words.append(word)

        word_counter = collections.Counter(current_words).most_common()
        word_dict = dict()
        word_dict[PAD_TOKEN] = 0
        word_dict[UNK_TOKEN] = 1
        word_dict[SOS_TOKEN] = 2
        word_dict[EOS_TOKEN] = 3

        # Consider words that have a minimum frequency
        eng_words = english_words()
        for word, count in word_counter:
            if count > min_frequency and (word in eng_words or word in string.punctuation):
                word_dict[word] = len(word_dict)

        # Adds words in dict <from_dict>
        if required_dict is not None:
            print("{} keys in <required_dict> parameter".format(len(required_dict)))
            added_words = 0
            for current_key in required_dict:
                if current_key not in word_dict:
                    word_dict[current_key] = len(word_dict)
                    added_words += 1
            print("{} words added to word_dict".format(added_words))

        file_utils.save(filename=output_file, data=word_dict)
    else:
        print("Loading word_dict from <{}>".format(output_file))
        word_dict = file_utils.load(filename=output_file)

    return word_dict


def build_word_dataset(path_to_data, word_dict, document_max_len):
    data_frame = pd.read_csv(path_to_data, names=["class", "title", "content"])

    punct_tokenize = partial(re.sub, pattern='([.,!?(){}]+)',
                             repl=' \g<1> ', flags=re.U)

    # Shuffle dataframe
    data_frame = data_frame.sample(frac=1)
    x = list(map(lambda d: word_tokenize(
        punct_tokenize(string=clean_str(d))), data_frame["content"]))
    x = list(map(lambda d: list(
        map(lambda w: word_dict.get(w, word_dict[UNK_TOKEN]), d)), x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d:
                 d + (document_max_len - len(d)) * [word_dict[PAD_TOKEN]], x))

    y = list(map(lambda d: d - 1, list(data_frame["class"])))

    return x, y


def filter_unk_tokens(data, max_percentage, word_dict):
    """
    Returns a list of indices of the list <data> to be removed if they have a
    percentage of UNK tokens greater than or equal to lower_bound_percentage
    :param data: data (list) to be filtered
    :param max_percentage: maximum percentage of UNK tokens to be allowed in the text
    :param word_dict: word dictionary to get the id of the unknown (UNK) token
    :return: data without the elements filtered
    """
    list_to_be_removed = []
    for i, tokens in enumerate(data):
        tokens = data[i]

        if len(tokens) == 0:
            # Remove any empty text that may be generated
            list_to_be_removed.append(i)
        else:
            # Counts how many UNK tokens are in the list and adds them to a list
            counts = collections.Counter(tokens)[word_dict[UNK_TOKEN]]

            percent_unk = (counts * 100) / len(tokens)
            if percent_unk >= max_percentage:
                list_to_be_removed.append(i)

    # Remove elements
    original_size = len(data)
    for index in reversed(list_to_be_removed):
        data.pop(index)

    percent_removed = (len(list_to_be_removed) * 100) / original_size
    print(f"We removed {len(list_to_be_removed)}/{original_size} elements"
          f" ({percent_removed:.2f}%) that have too many UNK tokens")

    return data


def build_text_dataset(path_to_data, word_dict, max_percentage_unk=50.0):
    data_frame = pd.read_csv(path_to_data, names=["class", "title", "content"])

    # Shuffle dataframe
    data_frame = data_frame.sample(frac=1)
    # Tokenize words
    x = list(map(lambda d: wordpunct_tokenize(clean_str(d)), data_frame["content"]))
    # Maps words to ids of word_dict
    x = list(map(lambda d: list(
        map(lambda w: word_dict.get(w, word_dict[UNK_TOKEN]), d)), x))

    # Filter out texts that have too many UNK tokens
    x = filter_unk_tokens(
        data=x, max_percentage=max_percentage_unk, word_dict=word_dict)

    return x


def text2one_hot_vector(input_text, word_dict):
    # Tokenize words
    x = wordpunct_tokenize(clean_str(input_text))
    # Maps words to ids of word_dict
    x = list(map(lambda w: word_dict.get(w, word_dict[UNK_TOKEN]), x))

    return x


def pre_process_sentence(batch_sentence, word_dict):
    # Tokenize words
    pre_proc_sentence = list(
        map(lambda d: wordpunct_tokenize(clean_str(d)), batch_sentence))
    # Maps words to ids of word_dict
    pre_proc_sentence = list(map(lambda d: list(
        map(lambda w: word_dict.get(w, word_dict[UNK_TOKEN]), d)), pre_proc_sentence))

    # Pad sequences smaller than the longest sequence in the batch
    max_sequence_length = max(len(l) for l in pre_proc_sentence)
    pre_proc_sentence = list(map(
        lambda d: d + (max_sequence_length - len(d)) * [word_dict[PAD_TOKEN]], pre_proc_sentence))

    return pre_proc_sentence


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    print(num_batches_per_epoch)
    for _ in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def pos_process_tokens(batch_sentences):
    pos_processed_batch = []
    for sequence in batch_sentences:
        text_sequence = " ".join(sequence).replace(
            PAD_TOKEN, "").replace(EOS_TOKEN, "").strip()
        pos_processed_batch.append(text_sequence)

    return pos_processed_batch
