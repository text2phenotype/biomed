"""
Methods & class for reporting the train/test results from a model
"""
import os

import pandas as pd

# # # identify available backends
# import matplotlib
# # gui_env = ["MacOSX", 'TKAgg', 'GTKAgg', 'Qt4Agg', 'Agg']  # need matplotlib >=3.1 to use MacOSX safely
# gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'Agg']
# for gui in gui_env:
#     try:
#         matplotlib.use(gui, warn=False, force=True)
#         from matplotlib import pyplot as plt
#         print(f"Using matplotlib backend: {gui}")
#         break
#     except Exception:
#         continue
import matplotlib.pyplot as plt

from biomed import RESULTS_PATH
from biomed.models.callbacks import TimerCallback

# TODO(mjp): make this backend selection smarter, defaulting to "Agg" if another backend doesn't work
# NOTE: we set the backend to be "Agg" here, which never creates figures on the GUI.
# To view figures (eg with plt.show()), you must set an appropriate backend for your system
# For modern MacOS systems, this could be "MacOSX". TKAgg also works on most systems
plt.switch_backend("Agg")


class ModelReportPlots:
    def __init__(self, model_name, job_id):
        self.model_name = model_name
        self.job_id = job_id
        self.output_path = os.path.join(RESULTS_PATH, self.job_id)
        os.makedirs(self.output_path, exist_ok=True)

    def plot_train_metrics(self, train_history: dict) -> None:
        """
        Plot the time series of the model train/validation history over all given metrics

        :param train_history: dict
            This dict can be loaded from train_history_{model_name}.json, in the model results folder
        :return: None
        """
        # convert json to dataframe
        df = pd.DataFrame(train_history)
        plot_style = '.-'

        skip_metrics = [TimerCallback.AVG_EPOCH_DUR_KEY, TimerCallback.TRAIN_DUR_KEY]

        metric_names = [col for col in df.columns if not col.startswith("val_") and col not in skip_metrics]

        for metric in metric_names:
            metric_names = [metric]
            if f"val_{metric}" in df.columns:
                metric_names += [f"val_{metric}"]
            df[metric_names].plot(style=plot_style)
            plt.legend()
            plt.xlabel("epoch")
            plt.ylabel(f"{metric}")
            plt.title(f"{metric} - {self.model_name}")
            plt.savefig(os.path.join(self.output_path, f"{self.model_name}_{metric}.png"))
