# -*- coding: utf-8 -*-
#refer solarisailab.com/arcguves/346
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

# 정수 형태의 node ID를 인간이 이해할 수 있는 레이블로 변환



class NodeLookup(object):
    def __init__(self, label_lookup_path=None, uid_lookup_path=None):
        if not label_lookup_path:  
            label_lookup_path = os.path.join(FLAGS.model_dir, 'imagenet_2012_challenge_label_map_proto.pbtxt')
        if not uid_lookup_path:
            uid_lookup_path = os.path.join(
                FLAGS.model_dir, 'imagenet_synset_to_human_label_map.txt')
            self.node_lookup = self.load(label_lookup_path, uid_lookup_path)
         
    def load(self, label_lookup_path, uid_lookup_path):       
        """각각의 softmax node에 대해 인간이 읽을 수 있는 영어 단어를 로드 함"
        Args:
            label_lookup_path : 정수 node ID 에 대한 문자 UID
            uid_lookup_path : 인간이 읽을 수 있는 문자에 대한 문자 UID
            
        Returns:
            정수 node ID 로부터 인간이 읽을 수 있는 문자에 대한 dict
        """
        if not tf.gfile.Exists(uid_lookup_path):
            tf.logging.fatal('File does not exist ')
        if not tf.gfile.Exists(label_lookup_path):
            tf.logging.fatal('File does not exist %s', label_lookup_path)  
            
        # 문자 UID로부터 인간이 읽을 수 있는 문자로의 맵핑을 로드함
        proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
        uid_to_human = {}
        p = re.compile(r'[n\d]*[\s,]*')
        
        for line in proto_as_ascii_lines:
            parsed_items = p.findall(line)
            uid = parsed_items[0]
            human_string = parsed_items[2]
            uid_to_human[uid] = human_string
        
        # 문자 UID 로부터 정수 node ID 데 대한 맴핑을 로드함
        
        
