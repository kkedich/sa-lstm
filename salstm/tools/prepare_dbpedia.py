"""
Prepares data of DBPedia and saves them into TFRecords for the sequence auto-encoder mode of
Dai and Le, Semi-supervised Sequence Learning
   - Pre-processing notes
   ---- punctuation is treated as separate tokens
   ---- any non-English characters and words in the DBpedia text are ignored.
         Note: keep in mind that removing non-English words can generate gaps in
         text that affect the semantic of the text. Thus, we remove non-english words,
         but we also remove entries that have too many UNK tokens (<max_percentage_unk>)
         after the pre-processing.
         Link to remove non-english words https://stackoverflow.com/a/41290205
   ---- words that only appear once are removed and we do not perform
         any term weighting or stemming.

References:
    - Link to download data: https://github.com/srhrshr/torchDatasets/


Example of usage:
   python3 -m salstm.tools.dataset.prepare_dbpedia \
              ./configs/conf_dbpedia.py


Example of conf_file.py
```
    # Optional
    required_dicts = [
        "./data/dictionary_words.json"
    ]
    dataset_files = dict(
      # Dbpedia .csv files
      train="./data/dbpedia/dbpedia_csv/train.csv",
      test="./data/dbpedia/dbpedia_csv/test.csv",
        # Output files
      records=dict(
        dir="records_dbpedia/",
        filename="records_dbpedia.tfrecord",
        train_files_list="train_record_tfrecord_files.txt",
        test_files_list="test_record_tfrecord_files.txt"
      )
    )
    working_dir="./output_files/dbpedia/dataset/"
```

Example of required dict:
 {'jogging': 0, 'running': 1, 'walking': 3}

"""

from argparse import ArgumentParser
from pathlib import Path

from salstm.datasets import text_record
from salstm.utils import config, dbpedia_utils, file_utils, misc


def parse_args():
    """Parse the input arguments"""
    parser = ArgumentParser(
        description="Script to prepare the text data of Dbpedia for training/evaluation.")
    parser.add_argument("conf_file", help="Configuration file for the dataset.")
    parser.add_argument(
        "--required_dict", type=str, default=None,
        help="File (.json) containing a pre-defined dictionary "
             "(key: word, value: integer value corresponding to the word index) "
             "to be considered when building the word dict to prevent these"
             "words to be mapped to the UNK token, if they are not in"
             "the dataset itself.")
    args = parser.parse_args()

    return args


def main():
    """Main function to prepare the DBPedia data."""
    args = parse_args()
    cfg_dataset = config.Config(args.conf_file)

    # Build word dictionary
    print("Building word dictionary..")

    required_dict = {}
    if cfg_dataset.required_dicts is not None:
        # Adds required dictionaries, since later these words will be used for
        # other models, and we cannot have them be mapped to UNK token.
        for required_dict_file in cfg_dataset.required_dicts:
            current_dict = file_utils.load_json(required_dict_file)
            required_dict = misc.merge_dicts(required_dict, current_dict)

    # Build dictionary of dbpedia
    Path(cfg_dataset.working_dir).mkdir(parents=True, exist_ok=True)

    word_dict = dbpedia_utils.build_word_dictionary(
        path_to_data=cfg_dataset.dataset_files.train,
        output_file=cfg_dataset.working_dir + "output_word_dictionary.pkl",
        required_dict=required_dict
    )
    print(f"{len(word_dict)} words in dictionary!")

    # Pre-processing of the dataset
    # Since we will use a sequence auto-encoder, we will not use the class
    # information of each text/document.
    # Generates one-hot vectors of the dataset (one hot in the sense that we
    # are using the word ids from the word dictionary)
    train_file = cfg_dataset.working_dir + "train_x.pkl"
    test_file = cfg_dataset.working_dir + "test_x.pkl"

    if not Path(cfg_dataset.working_dir + "train_x.pkl").exists():
        train_x = dbpedia_utils.build_text_dataset(
            cfg_dataset.dataset_files.train, word_dict, max_percentage_unk=45.0)
        test_x = dbpedia_utils.build_text_dataset(
            cfg_dataset.dataset_files.test, word_dict, max_percentage_unk=45.0)

        file_utils.save(train_file, train_x)
        file_utils.save(test_file, test_x)
        print(f"Train and test data saved at <{train_file}> and <{test_file}>, respectively\n")
    else:
        train_x = file_utils.load(train_file)
        test_x = file_utils.load(test_file)
        print(f"Train and test data loaded from <{train_file}> and <{test_file}>, respectively\n")
    print(f"{len(train_x)} documents in train_x and {len(test_x)} documents in test_x")

    # Save dataset into TFRecords
    dir_records = cfg_dataset.working_dir + cfg_dataset.dataset_files.records.dir
    Path(dir_records).mkdir(parents=True, exist_ok=True)

    tfrecord_files_train, _ = text_record.write(
        tfrecord_file=dir_records + cfg_dataset.dataset_files.records.filename,
        tag="train",
        text_data=train_x,
        num_shards=1
    )
    file_utils.save_list(dir_records + cfg_dataset.dataset_files.records.train_files_list,
                         tfrecord_files_train)

    tfrecord_files_test, _ = text_record.write(
        tfrecord_file=dir_records + cfg_dataset.dataset_files.records.filename,
        tag="test",
        text_data=test_x,
        num_shards=1
    )
    file_utils.save_list(dir_records + cfg_dataset.dataset_files.records.test_files_list,
                         tfrecord_files_test)

    print(f"The DBPedia dataset is ready now. Records were save at: <{dir_records}>")


if __name__ == '__main__':
    main()
