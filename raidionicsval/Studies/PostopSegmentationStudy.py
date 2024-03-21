from ..Studies.AbstractStudy import AbstractStudy
from ..Validation.validation_utilities import compute_fold_average
from ..Utils.resources import SharedResources
from ..Utils.io_converters import get_fold_from_file, reload_optimal_validation_parameters
import os
import pandas as pd
from pathlib import Path
import traceback

class PostopSegmentationStudy(AbstractStudy):

    def __init__(self):
        super().__init__()
        self.validation_folder = Path(self.input_folder, 'Validation')
        self.ext_validation_folder = Path(self.input_folder, 'ExternalValidation')
        self.interrater_validation_folder = Path(self.input_folder, 'InterraterStudy')

    def run(self):
        for c in self.class_names:
            self.format_results_latex_table(folder=self.validation_folder, class_name=c)
            self.format_results_latex_table(folder=self.ext_validation_folder, class_name=c)
            # super().compute_and_plot_overall(c, category='All')
            # super().compute_and_plot_overall(c, category='True Positive')

            # compute_fold_average(self.input_folder, class_optimal=self.classes_optimal, metrics=self.metric_names,
            #                      true_positive_state=False)
            # compute_fold_average(self.input_folder, class_optimal=self.classes_optimal, metrics=self.metric_names,
            #                      true_positive_state=True)

            # The 'GT volume (ml)' column is a default column computed during the validation phase
            # self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='PiW Dice',
            #                                                     metric2='GT volume (ml)', metric2_cutoffs=[1.],
            #                                                     category='All')

            # self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='HD95', metric2='GT volume (ml)',
            #                                                     metric2_cutoffs=[0.], category='All')
            # Other information, such as 'SpacZ', must be provided as part of the self.extra_patient_parameters
            # self.compute_and_plot_metric_over_metric_categories(class_name=c, metric1='PiW Dice', metric2='SpacZ',
            #                                                     metric2_cutoffs=[2.], category='All')

            # Correlation matrix between all metrics
            # super().compute_and_plot_metrics_correlation_matrix(class_name=c, category='All')
            # super().compute_and_plot_metrics_correlation_matrix(class_name=c, category='True Positive')

    def format_results_latex_table(self, folder, class_name,
                                   metrics=['PiW Dice_tp', 'HD95', 'VS', 'AVE', 'Patient-wise recall',
                                            'Patient-wise specificity', 'Patient-wise Balanced accuracy'], suffix=''):
        """
        :param folder: Destination folder where the results will be dumped (as specified in the configuration file)
        :param best_threshold:
        :param best_overlap:
        :param metric_names:
        :return:
        """
        try:
            results_filename = Path(folder, f"{class_name}_overall_metrics_average.csv")
            results = pd.read_csv(results_filename)

            results_tp_filename = Path(folder, f"{class_name}_overall_metrics_average_tp.csv")
            results_tp = pd.read_csv(results_tp_filename)

            fname = f'{class_name}_metrics_latex_table.txt' if suffix == '' else f'{class_name}_metrics_latex_table_{suffix}.txt'
            latex_table_fname = Path(folder, fname)
            output_string = ""

            for m in metrics:
                if m.endswith('_tp'):
                    res = results_tp.copy()
                    m = m.split('_')[0]
                else:
                    res = results.copy()

                mean = res.loc[0, m + ' (Mean)']
                std = res.loc[0, m + ' (Std)']

                if m not in ['HD95', 'AVE']:
                    mean, std = mean * 100, std * 100

                if std > 0:
                    output_string += f" & {mean:.2f}$\pm${std:.2f}"
                else:
                    output_string += f" & {mean:.2f}"

            pfile = open(latex_table_fname, 'w+')
            pfile.write(output_string + "\\\ \n")
            pfile.close()

        except Exception as e:
            print("Issue arose for class: {}.".format(class_name))
            print(traceback.format_exc())


