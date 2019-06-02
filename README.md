# botNet

### Table of Contents
* [About](#about)
* [Installation](#installation)
* [Todoist](#Todoist)
* [Running](#running)
  * [Chatbot](#chatbot)
* [Results](#results)
* [Pretrained model](#pretrained-model)
* [Improvements](#improvements)

## About

A chatbot based on _Transformer model with Attention_, proposed by [Google Brain Team](https://arxiv.org/pdf/1706.03762.pdf) for neural machine translation. The bot is mainly built with _Keras_ and _Tensorflow 2.0_ as backend.

- Dataset:  
  * [Cornell Movie Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
    (the files can be found in [data directory](data/cornell))

## Installation
Dependencies: `pip install -r requirements.txt`
  - python3 (tested with v3.6)
  - Tensorflow 2.0 (devbuild-20190529)
  - CUDA (find out more [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html))

## Todoist
-   [x] Finished processing database
-   [x] Creating training model based on [tensorflow/nmt](https://github.com/tensorflow/nmt)
-   [x] Arguments for better customization
-   [x] Config writer
-   [ ] Process more data from different sources (RC comments, Ubuntu corpus)
-   [ ] Fix TensorBoard
-   [ ] Document file

## Running

### Chatbot

To train the model, simply run `python main.py`. After training, run `python main.py --mode interactive` to interactive with the bot =)
Some useful tags for customization. For more options, run `python main.py -h`:
  * `--model_tag <name>`: allow to manage different models if you want to tweak with parameters
  * `--verbose`: print outputs for debugging purposes
  * `--continue_training`: continue training a saved model. Remember to include `--model_tag <name>` to keep training the designated model.
  * `--epochs <int>`: changing numbers of epochs for training.
  * `--num_layers <int>`: increasing # of layers for the network.

To visualize the computational graph and the cost with [TensorBoard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/), just run `tensorboard --logdir chatbot/save/model-<--model_tag>/logs/`. (_FIXING_)

This refers to [models.py](chatbot/model.py).

By default, the network architecture is a __stacked self-attention__ and __point-wise, fully connected layers__ for both encoder and decoder with the dimension size of 512 in 2 parallel layers. The network is trained using ADAM with custom learning rate. More about this can be found in [the paper](https://arxiv.org/pdf/1706.03762.pdf).

## Results

After 40 mins of training on GTX 1060M, the bot can respond correspondingly relatively well. One improvement is the increase dataset.

Here are some example response (more can be found [here](data/samples/output.txt)):

    Input: hello there
    Output: this is dr . smith s answering service .

    Input: what is your name?
    Output: nobody knows . there re my friend here .

    Input: name
    Output: yes sir .

    Input: is
    Output: what do you mean what ?

    Input: steve
    Output: an likely , herr mozart .

    Input: wow
    Output: let s go .

    Input: big comparison
    Output: think they re alright ?

    Input: they are
    Output: i suppose not , david . i have to stop and think of my dreams , d better get going .

    Input: i think i might train you more
    Output: is there two of us ? look at me !

    Input: good bye
    Output: oh , bill , i ll keep him all put this . i ll see side .

    Input: i will open the door
    Output: what ?

## Pretrained model

You can find _drunkboiv1_, a pretrained model [here](https://drive.google.com/file/d/1lA7EY-pIUx4_du3DA-R4ByfoqFeTBRHr/view?usp=sharing). To use it:
  1. Extract the zip inside the repository
  2. Copy the preprocessed dataset from `model-pretrainedv2/data/samples` to `data/samples/`.
  3. Copy pretrained weight from `model-pretrainedv2/save/` to `chatbot/save/`
  4. Run `python main.py --mode interactive`

## Improvements
 * More data I guess
 * Probably have to bring it to AWS for bigger params training
