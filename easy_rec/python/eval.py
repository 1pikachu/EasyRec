# -*- encoding:utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import logging
import os

import six
import json
import tensorflow as tf
from tensorflow.python.lib.io import file_io

from easy_rec.python.main import distribute_evaluate
from easy_rec.python.main import evaluate
from easy_rec.python.utils import config_util
from easy_rec.python.utils import ds_util

from easy_rec.python.utils.distribution_utils import set_tf_config_and_get_distribute_eval_worker_num_on_ds  # NOQA
if tf.__version__ >= '2.0':
  tf = tf.compat.v1

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s %(filename)s:%(lineno)d : %(message)s',
    level=logging.INFO)

tf.app.flags.DEFINE_string('pipeline_config_path', None,
                           'Path to pipeline config '
                           'file.')
tf.app.flags.DEFINE_string(
    'checkpoint_path', None, 'checkpoint to be evaled '
    ' if not specified, use the latest checkpoint in '
    'train_config.model_dir')
tf.app.flags.DEFINE_multi_string(
    'eval_input_path', None, 'eval data path, if specified will '
    'override pipeline_config.eval_input_path')
tf.app.flags.DEFINE_string('model_dir', None, help='will update the model_dir')
tf.app.flags.DEFINE_string('odps_config', None, help='odps config path')
tf.app.flags.DEFINE_string('edit_config_json', None, help='edit pipeline config str, example: {"model_dir":"experiments/",'
    '"feature_config.feature[0].boundaries":[4,5,6,7]}')
tf.app.flags.DEFINE_string('eval_result_path', 'eval_result.txt',
                           'eval result metric file')
tf.app.flags.DEFINE_bool('distribute_eval', False,
                         'use distribute parameter server for train and eval.')
tf.app.flags.DEFINE_bool('is_on_ds', False, help='is on ds')
tf.app.flags.DEFINE_integer('batch_size', 1, help='')
tf.app.flags.DEFINE_integer('num_examples', 200, help='')
tf.app.flags.DEFINE_bool('tensorboard', False, help='collect tensorboard')
FLAGS = tf.app.flags.FLAGS


def main(argv):
  if FLAGS.odps_config:
    os.environ['ODPS_CONFIG_FILE_PATH'] = FLAGS.odps_config

  if FLAGS.is_on_ds:
    ds_util.set_on_ds()
    if FLAGS.distribute_eval:
      set_tf_config_and_get_distribute_eval_worker_num_on_ds()

  assert FLAGS.model_dir or FLAGS.pipeline_config_path, 'At least one of model_dir and pipeline_config_path exists.'
  if FLAGS.model_dir:
    pipeline_config_path = os.path.join(FLAGS.model_dir, 'pipeline.config')
    if file_io.file_exists(pipeline_config_path):
      logging.info('update pipeline_config_path to %s' % pipeline_config_path)
    else:
      pipeline_config_path = FLAGS.pipeline_config_path
  else:
    pipeline_config_path = FLAGS.pipeline_config_path

  pipeline_config = config_util.get_configs_from_pipeline_file(
      pipeline_config_path)
  if FLAGS.model_dir:
    pipeline_config.model_dir = FLAGS.model_dir

  if FLAGS.distribute_eval:
    os.environ['distribute_eval'] = 'True'
    eval_result = distribute_evaluate(pipeline_config, FLAGS.checkpoint_path,
                                      FLAGS.eval_input_path,
                                      FLAGS.eval_result_path)
  else:
    os.environ['distribute_eval'] = 'False'
    if FLAGS.edit_config_json:
      pipeline_config = config_util.get_configs_from_pipeline_file(pipeline_config_path, False)
      edit_config_json = json.loads(FLAGS.edit_config_json)
      config_util.edit_config(pipeline_config, edit_config_json)
    else:
      pipeline_config.data_config.batch_size = FLAGS.batch_size
      pipeline_config.eval_config.num_examples = FLAGS.num_examples * FLAGS.batch_size
    eval_result = evaluate(pipeline_config, FLAGS.checkpoint_path,
                           FLAGS.eval_input_path, FLAGS.eval_result_path, FLAGS)
  if eval_result is not None:
    # when distribute evaluate, only master has eval_result.
    for key in sorted(eval_result):
      # skip logging binary data
      if isinstance(eval_result[key], six.binary_type):
        continue
      logging.info('%s: %s' % (key, str(eval_result[key])))
  else:
    logging.info('Eval result in master worker.')


if __name__ == '__main__':
  tf.app.run()
