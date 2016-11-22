# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# 필요한 라이브러리들을 임포트
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# classify_image_graph_def.pb:
#   GraphDef protocol buffer의 이진 표현
# imagenet_synset_to_human_label_map.txt:
#   synset ID를 인간이 읽을수 있는 문자로 맵핑
# imagenet_2012_challenge_label_map_proto.pbtxt:
#   protocol buffer의 문자 표현을 synset ID의 레이블로 맵핑

# Inception-v3 모델을 다운로드 받을 경로를 설정
tf.app.flags.DEFINE_string(
    'model_dir', '/tmp/imagenet',
    """Path to classify_image_graph_def.pb, """
    """imagenet_synset_to_human_label_map.txt, and """
    """imagenet_2012_challenge_label_map_proto.pbtxt.""")

# 읽을 이미지 파일의 경로를 설정
tf.app.flags.DEFINE_string('image_file', '',
                           """Absolute path to image file.""")
# 이미지의 추론결과를 몇개까지 표시할 것인지 설정
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")

# Inception-v3 모델을 다운로드할 URL 주소
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'



