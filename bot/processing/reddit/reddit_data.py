# Copyright 2019 Aaron Pham. All right reserved

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#   ==========================================================================
'''
    Descriptions: Data Preprocessing with tfds
        Corpus: Reddit Comment
'''
# import future
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import module
import os
import random  # Debugging purposes
import tensorflow_datasets as tfds  # Dataset processing because I'm kinda lazy
import tensorflow as tf
import bz2

class redditHandler:
    def __init__(self, args):
        self.args = args
