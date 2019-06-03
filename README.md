# botNet
[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)

## Disclaimer:
The information provided by the bot on the repository is for general study purposes only. All responses from the bot should not taken seriously. Under no circumstance shall I have any liability to you for any action as a result of the interaction with the bot. Thus, it is solely your responsibility for your own action.

## Table of Contents :satisfied:
* [About](#about)
* [Installation](#installation)
* [Todoist](#Todoist)
* [Running](#running)
  * [Chatbot](#chatbot)
  * [Interface](#interface)
* [Results](#results)
* [Pretrained model](#pretrained-model)
* [Improvements](#improvements)
* [Reference](#reference)

## About

A chatbot based on _Transformer model with Attention_, proposed by [Google Brain Team](https://arxiv.org/pdf/1706.03762.pdf) for neural machine translation. The bot is mainly built with _Keras_ and _Tensorflow 2.0_ as backend.

- Dataset:  
  * [Cornell Movie Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
    (the files can be found in [data](data/cornell))
  * [Reddit Comments](http://files.pushshift.io/reddit/comments/) (_UNDER WORKING_)

## Installation
Dependencies: `pip install -r requirements.txt`
  - python3 (tested with v3.6)
  - Tensorflow 2.0 (devbuild-20190529)
  - CUDA (find out more [here](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html))
  - django (built with 1.11)
  - Redis (more [here](https://redis.io/topics/quickstart))

## Todoist
-   [ ] Reddit data processing
-   [x] Website for deployment
-   [x] Fix TensorBoard
-   [x] Document file
-   [x] Automatically write configs for different models
-   [x] Arguments for better customization
-   [x] Creating training model based on [tensorflow/nmt](https://github.com/tensorflow/nmt)
-   [x] Processing Cornell Corpus

## Running

### Chatbot

To train the model, simply run `python main.py`. After training, run `python main.py --mode interactive` to interactive with the bot :smile: :monkey_face:

Some useful tags for customization. For more options, run `python main.py -h`:
  * `--model_tag <name>`: allow to manage different models if you want to tweak with parameters
  * `--verbose`: print outputs for debugging purposes
  * `--continue_training`: continue training a saved model. Remember to include `--model_tag <name>` to keep training the designated model.
  * `--epochs <int>`: changing numbers of epochs for training.
  * `--num_layers <int>`: increasing # of layers for the network.

To visualize the computational graph and the cost with [TensorBoard](https://www.tensorflow.org/how_tos/summaries_and_tensorboard/), just run `tensorboard --logdir chatbot/save/model-<--model_tag>/logs/`.

By default, the network architecture is a __stacked self-attention__ and __point-wise, fully connected layers__ for both encoder and decoder with the dimension size of 512 in 2 parallel layers. The network is trained using ADAM with custom learning rate. More about this can be found in [the paper](https://arxiv.org/pdf/1706.03762.pdf). This refers to [models.py](chatbot/model.py).


### Interface

I have created `server.bat` for Windows and `server.sh` for Linux. Such file can be used to run the server easily. The following part explains in detail how to manually startup the server.

  Once trained, there is an option for a more user friendly interface. The server will look at pretrained model `server` from the pretrained file (_you can train a model with --model_tag server_). For first time setup:

  - create a `misc.py` and put your own `BOT_SECRET_KEY='<your key here>'`

  ```bash
  cd botsite/
  python manage.py makemigrations
  python manage.py migrate
  ```

  Then launch the server locally with:

  ```bash
  cd botsite/
  redis-server & python manage.py runserver
  ```

  After launch, visit [localhost](http://localhost:8000/). More information can be found [here](https://docs.djangoproject.com/en/1.10/howto/deployment/checklist/).


## Results

After 105 mins of training on GTX 1060M, the bot can respond correspondingly relatively well. One improvement is the increase dataset.

Here are some example response :

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

You can find _pretrainedmodel-v2-with-server_ [here](https://drive.google.com/file/d/1vgOqA1Z-BAnaGh1NB9Gkt2envhai2U75/view?usp=sharing). To use it:
  1. Extract the zip inside the repository
  2. Copy the preprocessed dataset from `pretrainedmodel-v2-with-server/data/` to `data/`.
  3. Copy pretrained weight from `pretrainedmodel-v2-with-server/save/` to `save/`
  4. Run `python main.py --mode interactive`

## Improvements :bomb:
 * Data: More data should improve the capability of the bot. Such data can be increased from the [Reddit Comment](http://files.pushshift.io/reddit/comments/)
 * Increase training parameters: Increase `--num_layers`, `--d_model` for more complex architecture. However, alongside with increasing parameters, I might need better hardware.

## Reference
  * A big thanks toward [Etienne Pot](http://e-pot.xyz/) for his amazing work on [DeepQA](https://github.com/Conchylicultor/DeepQA) that inspire me to finish this project.
  * [Attention is All You Need](https://arxiv.org/abs/1706.03762) by _Google Brain Team_
