# SA-LSTM

This repository contains the implementation of the Sequence Auto-Encoder (SA-LSTM) from the paper of Dai and Le,
    ["Semi-supervised Sequence Learning"](https://arxiv.org/abs/1511.01432)
    
**Goal:** The goal of this implementation was to learn how the core of LSTMs works 
by using the [tf.nn.raw_rnn](https://github.com/tensorflow/docs/blob/r1.13/site/en/api_docs/python/tf/nn/raw_rnn.md) 
function of Tensorflow.
This function allow us to manage the inputs and outputs of each step of the Recurrent Neural Network.

Some highlights and notes:
+ Implementation of the SA-LSTM model using `tf.nn.raw_rnn`
+ The original paper of Dai and Le use a classic LSTM, here we can enable a bidirectional LSTM too.
+ The encoder deals with SOS and EOS tokens, so the inputs must only contain the text.
+ We also report the METEOR metric 

[//]: <> (How raw-rnn works?.)

### Requirements

+ Tensorflow 1.13.2
+ NLTK

You can find a Dockerfile available [here](docker/salstm.Dockerfile). 
For more information on how to run it check [these instructions](https://github.com/kkedich/docker-tensorflow-py3).
You basically need to create the image with `make build-image` and run the container with `make run-container`.

### Preparing the dataset

First, download the dataset from [here](https://github.com/srhrshr/torchDatasets/). Then, set a configuration file
to prepare the dataset (e.g., [example here](./configs/conf_dbpedia.py)). This will pre-process the dataset and save 
 the texts into a TFRecord file. Finally, execute the script 
`salstm/tools/prepare_dbpedia` as in:

```
   python3 -m salstm.tools.prepare_dbpedia \
              ./configs/conf_dbpedia.py
```


### Training the model

**Configuration files**

We use `.py` configuration files for each experiment. Parameters related to the text model, training, validation
settings, and the dataset are defined in this file. You can find an example in `./configs/`


**Metrics**:

- [Meteor](http://www.cs.cmu.edu/~alavie/METEOR/README.html): the training script will automatically download the metric
and run the corresponding command to obtain the score.


### Inference phase


### Current known issues and TODO list

+ Fix some issues with pylint and flake8
+ Update to newer versions of TF


### References

This is a list of some references used when working on this project:

+ [Tensorflow implementation of SA-LSTM](https://github.com/tensorflow/models/tree/master/research/adversarial_text) (first author of the paper is a contributor)
+ https://github.com/dongjun-Lee/transfer-learning-text-tf
+ https://github.com/isohrab/semi-supervised-text-classification
+ https://github.com/JayParks/tf-seq2seq/blob/master/seq2seq_model.py
+ https://github.com/tensorflow/nmt