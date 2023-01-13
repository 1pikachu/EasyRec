"""Hook that counts examples per second every N steps or seconds."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import tensorflow as tf
from tensorflow.python.training import session_run_hook


class ExamplesPerSecondHook(tf.estimator.SessionRunHook):
    """Hook to print out examples per second.

    Total time is tracked and then divided by the total number of steps
    to get the average step time and then batch_size is used to determine
    the running average of examples per second. The examples per second for the
    most recent interval is also logged.
    """

    def __init__(self,
                 batch_size,
                 every_n_steps=None,
                 every_n_secs=None,
                 warm_steps=0):
        """Initializer for ExamplesPerSecondHook.

        Args:
          batch_size: Total batch size across all workers used to calculate
            examples/second from global time.
          every_n_steps: Log stats every n steps.
          every_n_secs: Log stats every n seconds. Exactly one of the
            `every_n_steps` or `every_n_secs` should be set.
          warm_steps: The number of steps to be skipped before logging and running
            average calculation. warm_steps steps refers to global steps across all
            workers, not on each worker

        Raises:
          ValueError: if neither `every_n_steps` or `every_n_secs` is set, or
          both are set.
        """

        if (every_n_steps is None) == (every_n_secs is None):
          raise ValueError('exactly one of every_n_steps'
                           ' and every_n_secs should be provided.')

        self._timer = tf.estimator.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._step_train_time = 0
        self._total_steps = 0
        self._total_measured_steps = 0
        self._batch_size = batch_size
        self._warm_steps = warm_steps

    def begin(self):
        """Called once before using the session to check global step."""
        self._global_step_tensor = tf.compat.v1.train.get_global_step()
        if self._global_step_tensor is None:
          raise RuntimeError(
              'Global step should be created to use StepCounterHook.')

    def end(self, session):
        print("---------training end---------")
        print("batch_size: ", self._batch_size)
        print("total measured steps: ", self._total_measured_steps)
        print("total train time: ", self._step_train_time)
        average_examples_per_sec = self._batch_size * (
            self._total_measured_steps / self._step_train_time)
        print("training Throughput:", average_examples_per_sec)

    def before_run(self, run_context):  # pylint: disable=unused-argument
        """Called before each call to run().

        Args:
          run_context: A SessionRunContext object.

        Returns:
          A SessionRunArgs object or None if never triggered.
        """
        return tf.estimator.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
        """Called after each call to run().

        Args:
          run_context: A SessionRunContext object.
          run_values: A SessionRunValues object.
        """
        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
          elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(global_step)
          if elapsed_time is not None:
            # current examples per second is based on the elapsed training steps
            # and training time per batch
            current_examples_per_sec = self._batch_size * (
                elapsed_steps / elapsed_time)
            self._total_steps += elapsed_steps
            if global_step > self._warm_steps:
              self._step_train_time += elapsed_time
              self._total_measured_steps += elapsed_steps
              # average examples per second is based on the total (accumulative)
              # training steps and training time so far
              average_examples_per_sec = self._batch_size * (
                  self._total_measured_steps / self._step_train_time)
              # Current examples/sec followed by average examples/sec
              tf.compat.v1.logging.info('Batch [%g]:  last %g steps exp/sec = %g, total average exp/sec = '
                              '%g', self._total_steps, elapsed_steps, current_examples_per_sec, average_examples_per_sec)
            else:
              # Current examples/sec followed by completed warmup steps
              tf.compat.v1.logging.info('Batch [%g]:  last %g steps exp/sec = %g, completed %g/%g warmup steps'
                              , self._total_steps, elapsed_steps, current_examples_per_sec, self._total_steps, self._warm_steps)


class EvalHook(session_run_hook.SessionRunHook):
    def __init__(self, batch_size, warm_steps=20):
        self._step_eval_time = 0
        self._total_steps = 0
        self._total_measured_steps = 0
        self._batch_size = batch_size
        self._warm_steps = warm_steps
        
    def end(self, session):
        print("---------inference end---------")
        print("batch_size: ", self._batch_size)
        print("total measured steps: ", self._total_measured_steps)
        print("total inference time: ", self._step_eval_time)
        average_examples_per_sec = self._batch_size * (
            self._total_measured_steps / self._step_eval_time)
        print("inference Throughput:", average_examples_per_sec)
    
    def before_run(self, run_context):
        self.start_time = time.time()
    
    def after_run(self, run_context, run_values):
        duration = time.time() - self.start_time
        if self._total_steps >= self._warm_steps:
            self._total_measured_steps += 1
            self._step_eval_time += duration
        print("Iteration: {}, inference time: {}".format(self._total_steps, duration))
        self._total_steps += 1

