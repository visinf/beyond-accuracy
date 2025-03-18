# coding=utf-8
# Copyright 2023 The Robustness Metrics Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The base Metric class and an accuracy metric."""

import abc
import operator
from typing import Dict, Optional, List, Text, Dict

import numpy as np

class ModelPredictions:
  """Holds the predictions of a model on a specific dataset example.

  Properties:
    predictions: A list of predictions made on this example, each represented as
      a list of floats.
    time_in_s: The time in seconds the model took to make the predictions.
  """
  predictions: List[List[float]]
  time_in_s: Optional[float] = None


class Metric(metaclass=abc.ABCMeta):
  """The abstract class representing a metric.

  Each metric receives a set of predictions via calls to the `add_predictions`
  method, and computes the results using the `result` method.

  The function `result` might be called several times, and the results should
  be always computed using the data received over the complete life-time of
  the object.
  """

  def __init__(self, dataset_info=None):
    """Initializes the metric.

    Args:
      dataset_info: A datasets.DatasetInfo object.
    """

  @abc.abstractmethod
  def add_predictions(self,
                      model_predictions: ModelPredictions,
                      metadata) -> None:
    """Adds a new prediction that will be used when computing the metric.

    Multiple predictions for a single example can be added, but for adding
    predictions for multiple examples use `add_batch()`.

    Args:
      model_predictions: The predictions that the model made on an element
        from the dataset. Has an attribute `.predictions` with shape
        [num_predictions, ...] where [...] is the shape of a single prediction.
      metadata: The metadata for the example.
    """

  def add_batch(self,
                model_predictions,
                **metadata) -> None:
    """Adds a batch of predictions for a batch of examples.

    Args:
      model_predictions: The batch of predictions. Array with shape [batch_size,
        ...] where [...] is the shape of a single prediction. Some metric
        subclasses may require a shape [num_predictions, batch_size, ...] where
        they evaluate over multiple predictions per example.
      **metadata: Metadata for the batch of predictions. Each metadata kwarg,
        for example `label`, should be batched and have a leading axis of size
        `batch_size`.
    """
    def _recursive_map(fn, dict_or_val):
      if isinstance(dict_or_val, dict):
        return {k: _recursive_map(fn, v) for k, v in dict_or_val.items()}
      else:
        return fn(dict_or_val)

    for i, predictions_i in enumerate(np.array(model_predictions)):
      metadata_i = _recursive_map(operator.itemgetter(i), metadata)
      self.add_predictions(
          ModelPredictions(predictions=[predictions_i]),
          metadata_i)

  @abc.abstractmethod
  def result(self) -> Dict[Text, float]:
    """Computes the results from all the predictions it has seen so far.

    Returns:
      A dictionary mapping the name of each computed metric to its value.
    """


def _map_labelset(predictions, label, appearing_classes):
  """Indexes the predictions and label according to `appearing_classes`."""
  np_predictions = np.asarray(predictions)
  assert np_predictions.ndim == 2
  if appearing_classes:
    predictions = np_predictions[:, appearing_classes]
    predictions /= np.sum(predictions, axis=-1, keepdims=True)
    np_label = np.asarray(label)
    if np_label.ndim == 0:
      label = appearing_classes.index(label)
    else:
      assert np_label.ndim == 2
      label = np_label[:, appearing_classes]
  return predictions, label


class FullBatchMetric(Metric):
  """Base class for metrics that operate on the full dataset (not streaming)."""

  def __init__(self, dataset_info=None, use_dataset_labelset=False):
    self._ids_seen = set()
    self._predictions = []
    self._labels = []
    self._use_dataset_labelset = use_dataset_labelset
    self._appearing_classes = (dataset_info.appearing_classes if dataset_info
                               else None)
    super().__init__(dataset_info)

  def add_predictions(self, model_predictions, metadata) -> None:
    try:
      element_id = int(metadata["element_id"])
      if element_id in self._ids_seen:
        raise ValueError(f"You added element id {element_id!r} twice.")
      else:
        self._ids_seen.add(element_id)
    except KeyError:
      pass

    try:
      label = metadata["label"]
    except KeyError:
      raise ValueError("No labels in the metadata, provided fields: "
                       f"{metadata.keys()!r}")
    predictions = model_predictions.predictions
    if self._use_dataset_labelset:
      predictions, label = _map_labelset(
          predictions, label, self._appearing_classes)
    # If multiple predictions are present for a datapoint, average them:
    predictions = np.mean(predictions, axis=0)
    self._predictions.append(predictions)
    self._labels.append(label)

