# botNet

### Table of Contents
* [About](#about)
* [Installation](#installation)
* [Todoist](#Todoist)
* [Running](#running)
* [Results](#results)
* [Pretrained model](#pretrained-model)

## About

A chatbot based on [this paper](https://arxiv.org/pdf/1706.03762.pdf), proposing the _Transformer model with Attention_ for setence prediction. The bot is mainly built with Keras and Tensorflow 2.0 as backend.

- Supported Dataset:  
  * [Cornell Movie Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
    (the files can be found in [data directory](data/cornell))

## Installation
Dependencies: `pip install -r requirements.txt`
  - python3 (tested with v3.6)
  - Tensorflow 2.0 (devbuild-20190529)
  - CUDA (find out more [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html))
  -
## Todoist
-   [x] Finished processing database
-   [x] Creating training model based on [tensorflow/nmt](https://github.com/tensorflow/nmt)
-   [ ] Data Processing for more data from different sources (RC comments, Ubuntu corpus)
-   [ ] Arguments for better customization
-   [ ] Config writer
-   [ ] Fix Tensorboard

## Running

To train the model, simply edit the [config.py](chatbot/config.py) for your own customization. If you are not sure, just follow [the paper](https://arxiv.org/pdf/1706.03762.pdf). Then one can simply run `main.py` and type mode `train` for training new model.
After training, one can run `main.py` again and choose the option `test` to interact with the bot.

This refers to [models.py](chatbot/model.py)
By default, the network architecture is a stacked self-attention and point-wise, fully connected layers for both encoder and decoder with the dimension size of 512 with 2 parallel layers. The network is trained using ADAM with custom learning rate. More about this can be found in [the paper](https://arxiv.org/pdf/1706.03762.pdf).

## Results

After 3 hours of training with a 1060, the bot can respond correspondingly I guess. One of the main reasons is the dataset size.
Here are some example response (more can be found [here](data/samples/output.txt))

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

You can find [pretrained-modelv2](https://drive.google.com/file/d/13L_ndoAS7U2KVw1sS1H87rdyd4t_QXs8/view?usp=sharing) here, trained of the Cornell Corpus. To use it:
  1. Extract the zip inside the repository
  2. Copy the preprocessed dataset from `model-pretrainedv2/data/samples` to `data/samples/`.
  3. Copy pretrained weight from `model-pretrainedv2/save/models` to `save/models`
  4. Run `python main.py` and choose `test` options
