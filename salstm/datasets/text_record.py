"""
Code adapted from https://github.com/LightTag/BibSample/blob/master/preppy.py
"""
import math
import sys
from pathlib import Path

import tensorflow as tf


def sequence_to_tf_example(text_tokens):
    sequence_example = tf.train.SequenceExample()

    # A non-sequential feature of our example
    sequence_length = len(text_tokens)
    sequence_example.context.feature["sequence_length"].int64_list.value.append(sequence_length)

    # Feature lists for the two sequential features of our example
    fl_tokens = sequence_example.feature_lists.feature_list["tokens"]
    for token in text_tokens:
        fl_tokens.feature.add().int64_list.value.append(token)

    return sequence_example


def parse_text(sequence_example):
    context_features = {
        "sequence_length": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "tokens": tf.FixedLenSequenceFeature([], dtype=tf.int64),
    }

    # Parse the example (returns a dictionary of tensors)
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=sequence_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    return {"tokens": sequence_parsed["tokens"],
            "sequence_length": context_parsed["sequence_length"]}


def write(tfrecord_file, tag, text_data, num_shards=1):
    nb_sequences = len(text_data)
    texts_per_shard = int(math.ceil(nb_sequences / float(num_shards)))
    tfrecord_files = []
    count = 0

    file_path = Path(tfrecord_file)
    filename_without_suffix = file_path.parent / file_path.stem

    for shard_id in range(num_shards):
        output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
            filename_without_suffix, tag, shard_id, num_shards - 1)
        tfrecord_files.append(output_filename)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * texts_per_shard
            end_ndx = min((shard_id + 1) * texts_per_shard, nb_sequences)

            for current_text in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting text %d/%d, '
                                 'total_texts_in_shard=%d/total_text=%d, shard %d/%d'
                                 % (current_text + 1, end_ndx - 1,
                                    texts_per_shard, nb_sequences, shard_id + 1, num_shards))
                sys.stdout.flush()

                # Write text sequence
                example = sequence_to_tf_example(text_data[current_text])
                tfrecord_writer.write(example.SerializeToString())
                count += 1

    sys.stdout.write('\n')
    sys.stdout.flush()
    print(f"Summary: {count} records for {nb_sequences} sequences")

    return tfrecord_files, count
