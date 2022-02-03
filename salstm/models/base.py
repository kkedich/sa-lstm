
import tensorflow as tf
from salstm.utils import misc


class BaseModel:
    """ Base model that contains functions common to all models to be evaluated. """

    def __init__(self):
        self.scope_model = "None"
        self.metrics_values = None
        self.new_merged_summary_per_epoch_op = None

        # For the metrics part
        self.metrics_to_summarize = {}
        self.dict_quality_metrics = None

    def write_metrics_per_epoch(self, session, summary_writer, epoch):
        # Get the values of the metrics
        current_metrics = session.run(self.metrics_values)
        current_summary = session.run(self.new_merged_summary_per_epoch_op)
        print(f"current_metrics={current_metrics}")

        summary_writer.add_summary(current_summary, epoch)
        summary_writer.flush()

        current_metrics = misc.merge_dicts(current_metrics, {'epoch': epoch})

        return current_metrics

    def metrics_per_epoch(self):
        """ Build the metrics to be considered per epoch, and the merge summary operation
        to write them into Tensorboard summaries."""

        # Define the different metrics
        with tf.variable_scope(f"{self.scope_model}/metrics"):
            model_metrics = {}
            for key in self.metrics_to_summarize:
                model_metrics[f"{key}_per_epoch"] = tf.metrics.mean(
                    self.metrics_to_summarize[key])

            if self.dict_quality_metrics is not None:
                quality_metrics = {f"{k}_per_epoch": tf.metrics.mean(
                    v) for k, v in self.dict_quality_metrics.items()}
                metrics = misc.merge_dicts(model_metrics, quality_metrics)
            else:
                metrics = model_metrics

        # Group the update ops for the tf.metrics, so that we can run only one op to update them all
        self.update_metrics_op = tf.group(*[update_op for _, update_op in metrics.values()])
        self.metrics_values = {key: value[0] for key, value in metrics.items()}

        # Get the op to reset the local variables used in tf.metrics, for when we restart an epoch
        metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                             scope=f"{self.scope_model}/metrics")
        self.metrics_init_op = tf.variables_initializer(metric_variables)

        summaries_per_epoch = []
        for key, value in metrics.items():
            summaries_per_epoch.append(tf.summary.scalar(f"{self.scope_model}/{key}", value[0]))

        self.new_merged_summary_per_epoch_op = tf.summary.merge(summaries_per_epoch)

    def add_common_vars(self, additional_variables=None):
        """
        Add common variables to the Tensorboard summaries.
        """
        with tf.device('/cpu:0'):
            self.time_per_epoch = tf.Variable(0.0, name="time_per_epoch")

            summaries_per_epoch = [tf.summary.scalar("time_per_epoch", self.time_per_epoch)]
            if additional_variables is not None:
                for var_name in additional_variables:
                    summaries_per_epoch.append(tf.summary.scalar(
                        f"{var_name}_per_epoch", additional_variables[var_name]))

            merged_summary_per_epoch_op = tf.summary.merge(summaries_per_epoch)

        return merged_summary_per_epoch_op
