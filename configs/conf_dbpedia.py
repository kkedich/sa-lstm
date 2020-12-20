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
working_dir="./output_files/dbpedia/dataset_example/"
