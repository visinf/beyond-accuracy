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

"""Metrics that take into account the predicted uncertainty."""

from typing import Dict, Optional, Sequence, Union
import warnings

import numpy as np
#from robustness_metrics.datasets import base as datasets_base
#from robustness_metrics.metrics import base as metrics_base
from helper.base import Metric, FullBatchMetric



def _get_adaptive_bins(predictions, num_bins):
  """Returns upper edges for binning an equal number of datapoints per bin."""
  predictions = np.asarray(predictions).reshape(-1)
  if np.size(predictions) == 0:
    bin_upper_bounds = np.linspace(0, 1, num_bins + 1)[1:]
  else:
    edge_indices = np.linspace(
        0, np.size(predictions), num_bins, endpoint=False)

    # Round into integers for indexing. If num_bins does not evenly divide
    # len(predictions), this means that bin sizes will alternate between SIZE
    # and SIZE+1.
    edge_indices = np.round(edge_indices).astype(int)

    # If there are many more bins than data points, some indices will be
    # out-of-bounds by one. Set them to be within bounds:
    edge_indices = np.minimum(edge_indices, np.size(predictions) - 1)

    # Obtain the edge values:
    edges = np.sort(predictions)[edge_indices]

    # Following the convention of numpy.digitize, we do not include the leftmost
    # edge (i.e. return the upper bin edges):
    bin_upper_bounds = np.concatenate((edges[1:], [1.]))

  assert len(bin_upper_bounds) == num_bins and bin_upper_bounds[-1] == 1
  return bin_upper_bounds


def _binary_converter(probs):
  """Converts a binary probability vector into a matrix."""
  return np.array([[1 - p, p] for p in probs])


def _one_hot_encode(labels, num_classes=None):
  """One hot encoder for turning a vector of labels into a OHE matrix."""
  if num_classes is None:
    num_classes = len(np.unique(labels))
  return np.eye(num_classes)[labels]


def _is_monotonic(n_bins, bin_assign, labels):
  """Check if the label means in the bins are monotone.

  Args:
    n_bins: number of bins
    bin_assign: array/list of bin indices (int) assigning each example to bin.
    labels: array/list of class labels for each example in probs

  Returns:
    True if the provided bin_assign is monotonic.
  """
  bin_assign = np.array(bin_assign)
  last_ym = -1000
  for i in range(n_bins):
    cur = bin_assign == i
    if any(cur):
      ym = np.mean(labels[cur])
      if ym < last_ym:  # Determine if the predictions are monotonic.
        return False
      last_ym = ym
  return True


def _em_monotonic_sweep(probs, labels):
  """Compute bin assignments equal mass binning scheme."""
  probs = np.squeeze(probs)
  labels = np.squeeze(labels)
  probs = probs if probs.ndim > 0 else np.array([probs])
  labels = labels if labels.ndim > 0 else np.array([labels])

  sort_ix = np.argsort(probs)
  n_examples = len(probs)
  bin_assign = np.zeros((n_examples), dtype=int)

  prev_bin_assign = np.zeros((n_examples), dtype=int)
  for n_bins in range(2, n_examples):
    bin_assign[sort_ix] = np.minimum(
        n_bins - 1, np.floor(
            (np.arange(n_examples) / n_examples) * n_bins)).astype(int)
    if not _is_monotonic(n_bins, bin_assign, labels):
      return prev_bin_assign
    prev_bin_assign = np.copy(bin_assign)
  return bin_assign


def _ew_monotonic_sweep(probs, labels):
  """Monotonic bin sweep using equal width binning scheme."""
  n_examples = len(probs)
  bin_assign = np.zeros((n_examples), dtype=int)
  prev_bin_assign = np.zeros((n_examples), dtype=int)
  for n_bins in range(2, n_examples):
    bin_assign = np.minimum(n_bins - 1, np.floor(probs * n_bins)).astype(int)
    if not _is_monotonic(n_bins, bin_assign, labels):
      return prev_bin_assign
    prev_bin_assign = np.copy(bin_assign)
  return bin_assign


def _get_bin_edges(bin_assign, probs):
  """Convert bin_assign and probs to a set of bin_edges.

  Args:
    bin_assign: array/list of integer bin assignments.
    probs: array/list of corresponding probs to partition
      with bin_edges.

  Returns:
    bin_upper_bounds: array of right-side-edges that partition probs.

  Example:
  probs = [.2, .4, .6, .7, .9, .95]
  bin_assign = [0,0,1,1,2,2]

  bin_edges = get_bin_edges(bin_assign, probs)
  assert bin_edges == [.2, .5, .8, .95]

  Here bin_assign has 3 unique elements; therefore len(bin_edges) == 3+1
  min(bin_edges) == min(probs), and probs[0] == min(probs)
  max(bin_edges) == max(probs), and probs[-1] == max(probs)
  probs should be monotonically non-decreasing

  When an edge splits data, it does so by choosing the middle between
  the largest value in the left-bin and the smallest value in the right-bin.
  """

  bin_assign = np.squeeze(np.array(bin_assign))
  probs = np.squeeze(np.array(probs))
  bin_assign = bin_assign if bin_assign.ndim != 0 else np.array(
      [int(np.array(bin_assign))])
  probs = probs if probs.ndim != 0 else np.array([float(np.array(probs))])

  bin_edges = []
  curr_bin_max = None
  for ci, bin_ind in enumerate(set(bin_assign)):
    curr_bin_vals = probs[bin_assign == bin_ind]
    if len(curr_bin_vals) > 0:  # pylint: disable=g-explicit-length-test
      curr_bin_min = curr_bin_vals.min()
      curr_bin_max = curr_bin_vals.max()
      if ci == 0:
        bin_edges.append(curr_bin_min)
      else:
        bin_edges.append(curr_bin_min * .5 + previous_max * .5)  # pytype: disable=name-error
      previous_max = curr_bin_max
  if curr_bin_max is not None:
    bin_edges.append(curr_bin_max)

  # Validation relationships:
  if len(probs) > 0:  # pylint: disable=g-explicit-length-test
    assert bin_edges[-1] == max(bin_edges) == max(probs)
    assert bin_edges[0] == min(bin_edges) == min(probs)

  bin_upper_bounds = bin_edges[1:]
  return bin_upper_bounds


class _GeneralCalibrationErrorMetric:
  """Implements the space of calibration errors, General Calibration Error.

  For documentation of the parameters, see GeneralCalibrationError.
  """

  def __init__(self,
               binning_scheme,
               max_prob,
               class_conditional,
               norm,
               num_bins=30,
               threshold=0.0,
               datapoints_per_bin=None,
               distribution=None):
    self.binning_scheme = binning_scheme
    self.max_prob = max_prob
    self.class_conditional = class_conditional
    self.norm = norm
    self.num_bins = num_bins
    self.threshold = threshold
    self.datapoints_per_bin = datapoints_per_bin
    self.distribution = distribution
    self.accuracies = None
    self.confidences = None
    self.calibration_error = None
    self.calibration_errors = None

  def _get_mon_sweep_bins(self, probs, labels):
    """Adapter function to delegate bin_assign to the appropriate sweep method.

    Args:
      probs: array/list of corresponding probs to partition
        with bin_edges.
      labels: array/list of class labels for each example in probs
    Returns:
      Array of edges that partition the probabilities.
    """
    assert probs.ndim == 1
    assert labels.ndim == 1
    probs = probs[:, None]
    labels = labels[:, None]

    if self.binning_scheme == "adaptive":
      bin_assign = _em_monotonic_sweep(probs, labels)
    elif self.binning_scheme == "even":
      bin_assign = _ew_monotonic_sweep(probs, labels)
    else:
      raise NotImplementedError

    bin_edges = _get_bin_edges(bin_assign, probs)
    return bin_edges

  def _get_upper_bounds(self, probs_slice, labels):
    """Delegate construction of bin_upper_bounds to appropriate case-handler."""

    if self.binning_scheme == "adaptive" and self.num_bins is not None:
      bin_upper_bounds = _get_adaptive_bins(probs_slice, self.num_bins)
    elif self.binning_scheme == "adaptive" and self.num_bins is None:
      bin_upper_bounds = self._get_mon_sweep_bins(probs_slice, labels)
    elif self.binning_scheme == "even" and self.num_bins is None:
      bin_upper_bounds = self._get_mon_sweep_bins(probs_slice, labels)
    elif self.binning_scheme == "even" and self.num_bins is not None:
      bin_upper_bounds = np.histogram_bin_edges([],
                                                bins=self.num_bins,
                                                range=(0.0, 1.0))[1:]
    else:
      raise NotImplementedError(
          f"Condition not implemented: binning_scheme:{self.binning_scheme}, "
          f"num_bins:{self.num_bins}"
      )

    return bin_upper_bounds

  def _get_calibration_error(self, probs, labels, bin_upper_bounds):
    """Given a binning scheme, returns sum weighted calibration error."""
    probs = np.asarray(probs).reshape(-1)
    labels = np.asarray(labels).reshape(-1)

    if np.size(probs) == 0:
      return 0.

    bin_indices = np.digitize(probs, bin_upper_bounds)
    sums = np.bincount(bin_indices, weights=probs, minlength=self.num_bins)
    sums = sums.astype(np.float64)  # In case all probs are 0/1.
    counts = np.bincount(bin_indices, minlength=self.num_bins)
    counts = counts + np.finfo(sums.dtype).eps  # Avoid division by zero.
    self.confidences = sums / counts
    self.accuracies = np.bincount(
        bin_indices, weights=labels, minlength=self.num_bins) / counts

    self.calibration_errors = self.accuracies - self.confidences

    if self.norm == "l1":
      calibration_errors_normed = self.calibration_errors
    elif self.norm == "l2":
      calibration_errors_normed = np.square(self.calibration_errors)
    else:
      raise ValueError(f"Unknown norm: {self.norm}")

    weighting = counts / float(len(probs.flatten()))
    weighted_calibration_error = calibration_errors_normed * weighting

    return np.sum(np.abs(weighted_calibration_error))

  def update_state(self, labels, probs):
    """Updates the value of the General Calibration Error."""

    probs = np.array(probs)
    labels = np.array(labels)
    if probs.ndim == 2:

      num_classes = probs.shape[1]
      if num_classes == 1:
        probs = probs[:, 0]
        probs = _binary_converter(probs)
        num_classes = 2
    elif probs.ndim == 1:
      # Cover binary case
      probs = _binary_converter(probs)
      num_classes = 2
    else:
      raise ValueError("Probs must have 1 or 2 dimensions.")

    # Convert the labels vector into a one-hot-encoded matrix.

    labels_matrix = _one_hot_encode(labels, probs.shape[1])

    if self.datapoints_per_bin is not None:
      self.num_bins = int(len(probs) / self.datapoints_per_bin)
      if self.binning_scheme != "adaptive":
        raise ValueError(
            "To set datapoints_per_bin, binning_scheme must be 'adaptive'.")

    # When class_conditional is False, different classes are conflated.
    if not self.class_conditional:
      if self.max_prob:
        labels_matrix = labels_matrix[range(len(probs)),
                                      np.argmax(probs, axis=1)]
        probs = probs[range(len(probs)), np.argmax(probs, axis=1)]
      labels = np.squeeze(labels_matrix[probs > self.threshold])
      probs_slice = np.squeeze(probs[probs > self.threshold])
      bin_upper_bounds = self._get_upper_bounds(probs_slice, labels)
      calibration_error = self._get_calibration_error(probs_slice, labels,
                                                      bin_upper_bounds)

    # If class_conditional is true, predictions from different classes are
    # binned separately.
    else:
      # Initialize list for class calibration errors.
      class_calibration_error_list = []
      for j in range(num_classes):
        if not self.max_prob:
          probs_slice = probs[:, j]
          labels = labels_matrix[:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          bin_upper_bounds = self._get_upper_bounds(probs_slice, labels)

          calibration_error = self._get_calibration_error(
              probs_slice, labels, bin_upper_bounds)
          class_calibration_error_list.append(calibration_error / num_classes)
        else:
          # In the case where we use all datapoints,
          # max label has to be applied before class splitting.
          labels = labels_matrix[np.argmax(probs, axis=1) == j][:, j]
          probs_slice = probs[np.argmax(probs, axis=1) == j][:, j]
          labels = labels[probs_slice > self.threshold]
          probs_slice = probs_slice[probs_slice > self.threshold]
          bin_upper_bounds = self._get_upper_bounds(probs_slice, labels)
          calibration_error = self._get_calibration_error(
              probs_slice, labels, bin_upper_bounds)
          class_calibration_error_list.append(calibration_error / num_classes)
      calibration_error = np.sum(class_calibration_error_list)

    if self.norm == "l2":
      calibration_error = np.sqrt(calibration_error)

    self.calibration_error = calibration_error

  def result(self):
    return self.calibration_error

  def reset_states(self):
    self.calibration_error = None


class GeneralCalibrationError(FullBatchMetric):
  """Implements a large set of of calibration errors.

  This implementation of General Calibration Error can be class-conditional,
  adaptively binned, thresholded, focus on the maximum or top labels, and use
  the l1 or l2 norm. Can function as ECE, SCE, RMSCE, and more. For
  definitions of most of these terms, see [1].

  The metric returns a dict with keys:
    * "gce":  General Calibration Error. This is returned for all
        recalibration_method values, including None.
    * "beta": Optimal beta scaling, returned only for the temperature_scaling
        recalibration method

  Note that we also implement the following metrics by specializing this class
  and fixing some of its parameters:

  Static Calibration Error [1], registered under name "sce":
    binning_scheme="even"
    class_conditional=False
    max_prob=False
    norm="l1"

  Root Mean Squared Calibration Error [3], registered under "rmsce":
    binning_scheme="adaptive"
    class_conditional=False
    max_prob=True
    norm="l2"
    datapoints_per_bin=100

  Adaptive Calibration Error [1], registered under "ace":
    binning_scheme="adaptive"
    class_conditional=True
    max_prob=False
    norm="l1"

  Thresholded Adaptive Calibration Error [1], registered under "tace":
    binning_scheme="adaptive"
    class_conditional=True
    max_prob=False
    norm="l1"
    threshold=0.01

  Monotonic Sweep Calibration Error [4], registered under "msce":
    binning_scheme="adaptive"
    class_conditional=False
    max_prob=True
    norm="l1"
    num_bins=None

  ### References

  [1] Nixon, Jeremy, Michael W. Dusenberry, Linchuan Zhang, Ghassen Jerfel,
  and Dustin Tran. "Measuring Calibration in Deep Learning." In Proceedings of
  the IEEE Conference on Computer Vision and Pattern Recognition Workshops,
  pp. 38-41. 2019.
  https://arxiv.org/abs/1904.01685

  [2] Naeini, Mahdi Pakdaman, Gregory Cooper, and Milos Hauskrecht.
  "Obtaining well calibrated probabilities using bayesian binning."
  Twenty-Ninth AAAI Conference on Artificial Intelligence. 2015.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090/

  [3] Khanh Nguyen and Brendan O’Connor.
  "Posterior calibration and exploratory analysis for natural language
  processing models."  Empirical Methods in Natural Language Processing. 2015.
  https://arxiv.org/pdf/1508.05154.pdf

  [4] Rebecca Roelofs, Nicholas Cain, Jonathon Shlens, Michael C. Mozer
  "Mitigating bias in calibration error estimation."
  https://arxiv.org/pdf/2012.08668.pdf
  """

  def __init__(
      self,
      binning_scheme: str,
      max_prob: bool,
      class_conditional: bool,
      norm: str,
      num_bins: Optional[int],
      threshold: float,
      datapoints_per_bin: Optional[int] = None,
      fit_on_percent: float = 100.0,
      recalibration_method: Optional[str] = None,
      seed: Optional[int] = None,
      use_dataset_labelset: bool = False,
  ):
    """Initializes the GCE metric.

    If neither num_bins nor datapoints_per_bin are set, the bins are set using
    the monotone strategy in [4].

    Args:
      dataset_info: A datasets.DatasetInfo object.
      binning_scheme: Either "even" (for even spacing) or "adaptive"
        (for an equal number of datapoints in each bin).
      max_prob: "True" to measure calibration only on the maximum
        prediction for each datapoint, "False" to look at all predictions.
      class_conditional: "False" for the case where predictions from
        different classes are binned together, "True" for binned separately.
      norm: Apply "l1" or "l2" norm to the calibration error.
      num_bins: Number of bins of confidence scores to use.
      threshold: Ignore predictions below this value.
      datapoints_per_bin: When using an adaptive binning scheme, this determines
        the number of datapoints in each bin.
      fit_on_percent: Percentage of data used to fit recalibration function.
      recalibration_method: Takes values "temperature_scaling",
        "isotonic_regression" and None.
      seed: Randomness seed used for data shuffling before recalibration split.
      use_dataset_labelset: If set, and the given dataset has only a subset of
        the clases the model produces, the classes that are not in the dataset
        will be removed and the others scaled to sum up to one.
    """

    self._ids_seen = set()
    self._predictions = []
    self._labels = []
    self._eval_predictions = []
    self._eval_labels = []
    self._fit_predictions = []
    self._fit_labels = []

    self._binning_scheme = binning_scheme
    self._max_prob = max_prob
    self._class_conditional = class_conditional
    self._norm = norm
    self._num_bins = num_bins
    self._threshold = threshold
    self._datapoints_per_bin = datapoints_per_bin
    self._fit_on_percent = fit_on_percent
    self._seed = seed
    if not 0 <= fit_on_percent <= 100:
      raise ValueError(f"Argument fit_on_percent={fit_on_percent} is not within"
                       " expected range [0,100].")
    self._recalibration_method = recalibration_method
    if fit_on_percent == 100.0 and recalibration_method is not None:
      warnings.warn("Recalibration without data split: You are both fitting and"
                    " rescaling on the entire data set (method: "
                    f"{recalibration_method}). Set 'fit_on_percent'<100 or "
                    "recalibration_method=None.")
    if fit_on_percent == 0.0 and recalibration_method is not None:
      warnings.warn("No recalibration without fitting data: You selected the "
                    f"recalibration method {recalibration_method} and specified"
                    f" fitting on {fit_on_percent} percent of the data. "
                    " Recalibration is skipped. Set 'fit_on_percent'>0 to "
                    "recalibrate data with selected method.")

    super().__init__(use_dataset_labelset=use_dataset_labelset)

  def result(self) -> Dict[str, float]:
    self.shuffle_and_split_data()

    m = _GeneralCalibrationErrorMetric(
        binning_scheme=self._binning_scheme,
        max_prob=self._max_prob,
        class_conditional=self._class_conditional,
        norm=self._norm,
        num_bins=self._num_bins,
        threshold=self._threshold,
        datapoints_per_bin=self._datapoints_per_bin,
    )
    m.update_state(np.asarray(self._eval_labels),
                   np.asarray(self._eval_predictions))

    return {"gce": m.result()}

  def shuffle_and_split_data(self) -> None:
    n_labels = len(self._labels)
    number_of_fit_examples = round(self._fit_on_percent*0.01*n_labels)
    labels = np.asarray(self._labels)
    predictions = np.asarray(self._predictions)
    if number_of_fit_examples == n_labels:
      # No shuffling, no data split.
      # Fit and evaluate on the (same) complete data set.
      self._eval_predictions = predictions
      self._eval_labels = labels
      self._fit_predictions = predictions
      self._fit_labels = labels
    else:
      # Shuffle ordered pair (labels, predictions) by using the permutation
      # method of the random number generator.
      # After updating to numpy 1.17 : use
      # rng = np.random.default_rng(seed=self._seed)
      # perm = rng.permutation(len(lbls))
      perm = np.random.RandomState(seed=self._seed).permutation(n_labels)
      labels, predictions = labels[perm], predictions[perm]
      self._eval_predictions = predictions[number_of_fit_examples:]
      self._eval_labels = labels[number_of_fit_examples:]
      self._fit_predictions = predictions[:number_of_fit_examples]
      self._fit_labels = labels[:number_of_fit_examples]


class RootMeanSquaredCalibrationError(GeneralCalibrationError):

  def __init__(self,
               num_bins: int = 30,
               **kwargs):
    super().__init__(
                     threshold=0,
                     binning_scheme="adaptive",
                     max_prob=True,
                     class_conditional=False,
                     norm="l2",
                     num_bins=num_bins,
                     **kwargs)


class StaticCalibrationError(GeneralCalibrationError):

  def __init__(self,
               num_bins: int = 30,
               **kwargs):
    super().__init__(
                     threshold=0,
                     binning_scheme="even",
                     max_prob=False,
                     class_conditional=True,
                     norm="l1",
                     num_bins=num_bins,
                     **kwargs)

class AdaptiveCalibrationError(GeneralCalibrationError):

  def __init__(self,
               num_bins: int = 30,
               **kwargs):
    super().__init__(
                     threshold=0,
                     binning_scheme="adaptive",
                     max_prob=False,
                     class_conditional=True,
                     norm="l1",
                     num_bins=num_bins,
                     **kwargs)

class ThresholdedAdaptiveCalibrationError(GeneralCalibrationError):

  def __init__(self,
               num_bins: int = 30,
               threshold: float = 0.01,
               **kwargs):
    super().__init__(
                     threshold=0,
                     binning_scheme="adaptive",
                     max_prob=False,
                     class_conditional=True,
                     norm="l1",
                     num_bins=num_bins,
                     **kwargs)

class MonotonicSweepCalibrationError(GeneralCalibrationError):

  def __init__(self,
               num_bins: int = 30,
               **kwargs):
    super().__init__(
                     threshold=0,
                     binning_scheme="adaptive",
                     class_conditional=False,
                     max_prob=True,
                     norm="l1",
                     num_bins=None,
                     **kwargs)
