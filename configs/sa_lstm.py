text_model = dict(
    type = "sa_lstm",
    word_embedding_size = 128, # 256
    num_hidden_units = 64,
    dropout_rate_embeddings = 0.5,
    dropout_rate_output = 0.5,
    bidirectional_encoder=False, # True
    swap_memory=False,
    optimizer=dict(
        learning_rate=0.001,
        beta1=0.5,
        type_optimizer="adam",
        max_gradient_norm = 5.0
    )
)
train_cfg = dict(
    epochs=100,
    batch_size=16,
    display_loss_every_step=50,
    save_summary_every_step=150,
    save_model_every_epoch=10
)
val_cfg = dict(
    display_loss_every_step=50,
    save_summary_every_step=100
)
# Dataset settings
dataset_cfg = dict(
    name = "dbpedia",
    records=dict(
        root="./output_files/dbpedia/dataset/records_dbpedia/",
        train="train_record_tfrecord_files.txt",
        val="val_record_tfrecord_files.txt",
        test="test_record_tfrecord_files.txt"
    ),
    word_dict = "./output_files/dbpedia/dataset/word_dictionary.pkl",
    original_train_file = "./data/dbpedia/dbpedia_csv/train.csv",
    original_test_file = "./data/dbpedia/dbpedia_csv/test.csv"
)
output_dir="./output_files/dbpedia/models_sa_lstm/"
meteor_dir="./data/meteor/meteor-1.5/"