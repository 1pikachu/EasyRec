# -*- encoding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
from collections import OrderedDict

import tensorflow as tf

from easy_rec.python.compat import regularizers
from easy_rec.python.compat.feature_column import feature_column
from easy_rec.python.feature_column.feature_column import FeatureColumnParser
from easy_rec.python.feature_column.feature_group import FeatureGroup
from easy_rec.python.layers import sequence_feature_layer
from easy_rec.python.layers import variational_dropout_layer
from easy_rec.python.layers.common_layers import text_cnn
from easy_rec.python.protos.feature_config_pb2 import WideOrDeep
from easy_rec.python.utils import shape_utils

from easy_rec.python.compat.feature_column.feature_column import _SharedEmbeddingColumn  # NOQA
from easy_rec.python.compat.feature_column.feature_column_v2 import EmbeddingColumn  # NOQA
if tf.__version__ >= '2.0':
  tf = tf.compat.v1


class InputLayer(object):
  """Input Layer for generate input features.

  This class apply feature_columns to input tensors to generate wide features and deep features.
  """

  def __init__(self,
               feature_configs,
               feature_groups_config,
               variational_dropout_config=None,
               wide_output_dim=-1,
               ev_params=None,
               embedding_regularizer=None,
               kernel_regularizer=None,
               is_training=False):
    self._feature_groups = {
        x.group_name: FeatureGroup(x) for x in feature_groups_config
    }
    self.sequence_feature_layer = sequence_feature_layer.SequenceFeatureLayer(
        feature_configs, feature_groups_config, ev_params,
        embedding_regularizer, kernel_regularizer, is_training)
    self._seq_feature_groups_config = []
    for x in feature_groups_config:
      for y in x.sequence_features:
        self._seq_feature_groups_config.append(y)
    self._group_name_to_seq_features = {
        x.group_name: x.sequence_features
        for x in feature_groups_config
        if len(x.sequence_features) > 0
    }
    wide_and_deep_dict = self.get_wide_deep_dict()
    self._fc_parser = FeatureColumnParser(
        feature_configs,
        wide_and_deep_dict,
        wide_output_dim,
        ev_params=ev_params)

    self._embedding_regularizer = embedding_regularizer
    self._kernel_regularizer = kernel_regularizer
    self._is_training = is_training
    self._variational_dropout_config = variational_dropout_config

  def has_group(self, group_name):
    return group_name in self._feature_groups

  def __call__(self, features, group_name, is_combine=True):
    """Get features by group_name.

    Args:
      features: input tensor dict
      group_name: feature_group name
      is_combine: whether to combine sequence features over the
          time dimension.

    Return:
      features: all features concatenate together
      group_features: list of features
      seq_features: list of sequence features, each element is a tuple:
          3 dimension embedding tensor (batch_size, max_seq_len, embedding_dimension),
          1 dimension sequence length tensor.
    """
    assert group_name in self._feature_groups, 'invalid group_name[%s], list: %s' % (
        group_name, ','.join([x for x in self._feature_groups]))
    feature_name_to_output_tensors = {}
    negative_sampler = self._feature_groups[group_name]._config.negative_sampler
    if group_name in self._group_name_to_seq_features:
      for seq_feature in self._group_name_to_seq_features[group_name]:
        for seq_att in seq_feature.seq_att_map:
          for k in seq_att.key:
            feature_name_to_output_tensors[k] = None
    if is_combine:
      concat_features, group_features = self.single_call_input_layer(
          features, group_name, is_combine, feature_name_to_output_tensors)
      if group_name in self._group_name_to_seq_features:
        concat_features, all_seq_fea = self.sequence_feature_layer(
            features,
            concat_features,
            self._group_name_to_seq_features[group_name],
            feature_name_to_output_tensors,
            negative_sampler=negative_sampler)
        group_features.extend(all_seq_fea)
        all_seq_fea = tf.concat(all_seq_fea, axis=-1)
        concat_features = tf.concat([concat_features, all_seq_fea], axis=-1)
      return concat_features, group_features
    else:
      if self._variational_dropout_config is not None:
        raise ValueError(
            'variational dropout is not supported in not combined mode now.')
      return self.single_call_input_layer(features, group_name, is_combine)

  def single_call_input_layer(self,
                              features,
                              group_name,
                              is_combine=True,
                              feature_name_to_output_tensors=None):
    """Get features by group_name.

    Args:
      features: input tensor dict
      group_name: feature_group name
      is_combine: whether to combine sequence features over the
          time dimension.
      feature_name_to_output_tensors: if set sequence_features, feature_name_to_output_tensors will
      take key tensors to reuse.

    Return:
      features: all features concatenate together
      group_features: list of features
      seq_features: list of sequence features, each element is a tuple:
          3 dimension embedding tensor (batch_size, max_seq_len, embedding_dimension),
          1 dimension sequence length tensor.
    """
    assert group_name in self._feature_groups, 'invalid group_name[%s], list: %s' % (
        group_name, ','.join([x for x in self._feature_groups]))
    feature_group = self._feature_groups[group_name]
    group_columns, group_seq_columns = feature_group.select_columns(
        self._fc_parser)
    if is_combine:
      cols_to_output_tensors = OrderedDict()
      output_features = feature_column.input_layer(
          features,
          group_columns,
          cols_to_output_tensors=cols_to_output_tensors,
          feature_name_to_output_tensors=feature_name_to_output_tensors)
      embedding_reg_lst = [output_features]
      builder = feature_column._LazyBuilder(features)
      seq_features = []
      for column in sorted(group_seq_columns, key=lambda x: x.name):
        with tf.variable_scope(None, default_name=column._var_scope_name):
          seq_feature, seq_len = column._get_sequence_dense_tensor(builder)
          embedding_reg_lst.append(seq_feature)

          sequence_combiner = column.sequence_combiner
          if sequence_combiner is None:
            raise ValueError(
                'sequence_combiner is none, please set sequence_combiner or use TagFeature'
            )
          if sequence_combiner.WhichOneof('combiner') == 'attention':
            attn_logits = tf.layers.dense(
                inputs=seq_feature,
                units=1,
                kernel_regularizer=self._kernel_regularizer,
                use_bias=False,
                activation=None,
                name='attention')
            attn_logits = tf.squeeze(attn_logits, axis=-1)
            attn_logits_padding = tf.ones_like(attn_logits) * (-2**32 + 1)
            seq_mask = tf.sequence_mask(seq_len)
            attn_score = tf.nn.softmax(
                tf.where(seq_mask, attn_logits, attn_logits_padding))
            seq_feature = tf.reduce_sum(
                attn_score[:, :, tf.newaxis] * seq_feature, axis=1)
            seq_features.append(seq_feature)
            cols_to_output_tensors[column] = seq_feature
          elif sequence_combiner.WhichOneof('combiner') == 'text_cnn':
            num_filters = sequence_combiner.text_cnn.num_filters
            filter_sizes = sequence_combiner.text_cnn.filter_sizes
            cnn_feature = text_cnn(seq_feature, filter_sizes, num_filters)
            seq_features.append(cnn_feature)
            cols_to_output_tensors[column] = cnn_feature
          else:
            raise NotImplementedError
      if self._variational_dropout_config is not None:
        features_dimension = OrderedDict([
            (k.raw_name, int(v.shape[-1]))
            for k, v in cols_to_output_tensors.items()
        ])
        concat_features = tf.concat([output_features] + seq_features, axis=-1)
        variational_dropout = variational_dropout_layer.VariationalDropoutLayer(
            self._variational_dropout_config,
            features_dimension,
            self._is_training,
            name=group_name)
        concat_features = variational_dropout(concat_features)
        group_features = tf.split(
            concat_features, list(features_dimension.values()), axis=-1)
      else:
        concat_features = tf.concat([output_features] + seq_features, axis=-1)
        group_features = [cols_to_output_tensors[x] for x in group_columns] + \
                         [cols_to_output_tensors[x] for x in group_seq_columns]

      regularizers.apply_regularization(
          self._embedding_regularizer, weights_list=embedding_reg_lst)
      return concat_features, group_features

    else:  # return sequence feature in raw format instead of combine them
      assert len(group_columns) == 0, \
          'there are none sequence columns: %s' % str(group_columns)
      builder = feature_column._LazyBuilder(features)
      seq_features = []
      embedding_reg_lst = []
      for fc in group_seq_columns:
        with tf.variable_scope('input_layer/' + fc.categorical_column.name):
          tmp_embedding, tmp_seq_len = fc._get_sequence_dense_tensor(builder)
          if fc.max_seq_length > 0:
            tmp_embedding, tmp_seq_len = shape_utils.truncate_sequence(
                tmp_embedding, tmp_seq_len, fc.max_seq_length)
          seq_features.append((tmp_embedding, tmp_seq_len))
          embedding_reg_lst.append(tmp_embedding)
      regularizers.apply_regularization(
          self._embedding_regularizer, weights_list=embedding_reg_lst)
      return seq_features

  def get_wide_deep_dict(self):
    """Get wide or deep indicator for feature columns.

    Returns:
      dict of { feature_name : WideOrDeep }
    """
    wide_and_deep_dict = {}
    for fg_name in self._feature_groups.keys():
      fg = self._feature_groups[fg_name]
      tmp_dict = fg.wide_and_deep_dict
      for k in tmp_dict:
        v = tmp_dict[k]
        if k not in wide_and_deep_dict:
          wide_and_deep_dict[k] = v
        elif wide_and_deep_dict[k] != v:
          wide_and_deep_dict[k] = WideOrDeep.WIDE_AND_DEEP
        else:
          pass
    return wide_and_deep_dict
