from ..Validation.validation_utilities import compute_fold_average
from ..Utils.resources import SharedResources
from ..Utils.io_converters import get_fold_from_file, reload_optimal_validation_parameters
import os
import pandas as pd
from pathlib import Path
import traceback
import nibabel as nib
from shutil import copyfile
import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import List
import traceback
from ..Validation.kfold_model_validation import ModelValidation
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy


class ComparePostopStudy():

    def __init__(self):
        self.input_folder = SharedResources.getInstance().studies_input_folder
        self.output_folder = SharedResources.getInstance().studies_output_folder
        self.metric_names = []

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.experiment_base_folder = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation')
        # self.experiments = ['exp1_patch_wise_150123', 'exp2_patch_wise_w_neg_190123', 'exp3_patch_wise_w_neg_scratch',
        #                     'exp4_pw_w_neg_pretrained_260124', 'exp5_patching_in_data_load_96x96x96',
        #                     'exp6_2levels_kernelsize7', 'exp7_kernelsize5', 'exp8_dilated_GT', 'exp9_center_around_tumor']
        self.experiments = ['exp1_patch_wise_150123', 'exp3_patch_wise_w_neg_scratch', 'exp4_pw_w_neg_pretrained_260124',
                            'exp5_patching_in_data_load_96x96x96', 'exp9_center_around_tumor',
                            'exp6_2levels_kernelsize7', 'exp7_kernelsize5']
        self.experiment_output_names = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6', 'exp7']

        self.alex_experiments_base_folder = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/old_experiments/Alexandros_experiments')
        self.alex_experiments = [f'90{str(i)}' for i in range(1, 6)]

        self.agunet_experiments_base_folder = Path(
            '/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/old_experiments/TRD_experiments')
        self.agunet_experiments = ['run2_exp2_T1c_T1w']

        self.rater_base_folder = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/InterraterValidation')
        self.annotator_groups = ['nov', 'exp']
        self.annotators = [f"{annot}{i}" for i in range(1, 5) for annot in self.annotator_groups]
        self.consensus_annotations = ['strict-consensus-nov', 'strict-consensus-exp', 'strict-consensus-all-annotators']
        self.all_annotators = self.consensus_annotations # + self.annotators

        self._class_names = SharedResources.getInstance().studies_class_names

    def run(self):
        # self.format_latex_table_external_test_results(metrics=['PiW Dice_tp', 'HD95', 'Patient-wise recall',
        #                                              'Patient-wise specificity', 'Patient-wise Balanced accuracy'],
        #                                               fname_suffix='external_test_results_latex_table_final')
        # self.format_latex_table_interrater_results(metrics=['PiW Dice_tp', 'HD95', 'Patient-wise recall',
        #                                              'Patient-wise specificity', 'Patient-wise Balanced accuracy'],
        #                                            fname_suffix='interrater_results_latex_table_final')

        volume_study_folders = [Path(self.experiment_base_folder, 'exp9_center_around_tumor', 'InterraterStudy'),
                                Path(self.rater_base_folder, 'strict-consensus-all-annotators', 'InterraterStudy'),
                                Path(self.rater_base_folder, 'all-union', 'InterraterStudy')
                                ]
        volume_study_names = ['exp9',  'Consensus annotation', 'Union annotation'] #'expert consensus',
        self.plot_volume(volume_study_folders, volume_study_names)

    @property
    def class_names(self) -> List[str]:
        return self._class_names

    def format_latex_table_external_test_results(self, metrics=['PiW Dice_tp', 'HD95', 'VS', 'AVE', 'Patient-wise recall',
                                                     'Patient-wise specificity', 'Patient-wise Balanced accuracy'],
                                                    fname_suffix='external_test_results_latex_table'):
        for c in self.class_names:
            output_fname = Path(self.output_folder, f'{c}_{fname_suffix}.txt')
            output_string = ""

            for i, exp in enumerate(self.experiments):
                results_folder = Path(self.experiment_base_folder, exp, 'ExternalValidation')
                # exp_name_latex = exp.replace('_', '\\_')
                exp_name_latex = self.experiment_output_names[i].replace('_', '\\_')
                output_str_exp = self._format_results_latex_table_one_exp(results_folder, c, metrics=metrics)
                output_string += exp_name_latex + " " + output_str_exp + '\n'

            output_string += "\\hline \n"
            for exp in self.alex_experiments:
                results_folder = Path(self.alex_experiments_base_folder, exp, 'ExternalValidation')
                exp_name_latex = exp.replace('_', '\\_')
                output_str_rater = self._format_results_latex_table_one_exp(results_folder, c, metrics=metrics)
                output_string += exp_name_latex + " " + output_str_rater + '\n'

            output_string += "\\hline \n"
            for exp in self.agunet_experiments:
                results_folder = Path(self.agunet_experiments_base_folder, exp, 'ExternalValidation')
                exp_name_latex = exp.replace('_', '\\_')
                output_str_rater = self._format_results_latex_table_one_exp(results_folder, c, metrics=metrics)
                output_string += exp_name_latex + " " + output_str_rater + '\n'

            pfile = open(output_fname, 'w+')
            pfile.write(output_string)
            pfile.close()
    def format_latex_table_interrater_results(self, metrics=['PiW Dice_tp', 'HD95', 'VS', 'AVE', 'Patient-wise recall',
                                                     'Patient-wise specificity', 'Patient-wise Balanced accuracy'],
                                              fname_suffix='interrater_results_latex_table'):
        for c in self.class_names:
            output_fname = Path(self.output_folder, f'{c}_{fname_suffix}.txt')
            output_string = ""

            for i, exp in enumerate(self.experiments):
                results_folder = Path(self.experiment_base_folder, exp, 'InterraterStudy')
                # exp_name_latex = exp.replace('_', '\\_')
                exp_name_latex = self.experiment_output_names[i].replace('_', '\\_')
                output_str_exp = self._format_results_latex_table_one_exp(results_folder, c, metrics=metrics)
                output_string += exp_name_latex + " " + output_str_exp + '\n'

            output_string += "\\hline \n"
            for exp in self.alex_experiments:
                results_folder = Path(self.alex_experiments_base_folder, exp, 'InterraterStudy')
                exp_name_latex = exp.replace('_', '\\_')
                output_str_rater = self._format_results_latex_table_one_exp(results_folder, c, metrics=metrics)
                output_string += exp_name_latex + " " + output_str_rater + '\n'

            output_string += "\\hline \n"
            for exp in self.agunet_experiments:
                results_folder = Path(self.agunet_experiments_base_folder, exp, 'InterraterStudy')
                exp_name_latex = exp.replace('_', '\\_')
                output_str_rater = self._format_results_latex_table_one_exp(results_folder, c, metrics=metrics)
                output_string += exp_name_latex + " " + output_str_rater + '\n'

            output_string += "\\hline \n"
            for rater in self.all_annotators:
                results_folder = Path(self.rater_base_folder, rater, 'InterraterStudy')
                rater_name_latex = rater.replace('_', '\\_')
                output_str_rater = self._format_results_latex_table_one_exp(results_folder, c, metrics=metrics)
                output_string += rater_name_latex + " " + output_str_rater + '\n'

            pfile = open(output_fname, 'w+')
            pfile.write(output_string)
            pfile.close()

    def _format_results_latex_table_one_exp(self, folder, class_name,
                                            metrics=['PiW Dice_tp', 'HD95', 'VS', 'AVE', 'Patient-wise recall',
                                                     'Patient-wise specificity', 'Patient-wise Balanced accuracy']):
        """
        :param folder: Destination folder where the results will be read
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

            output_string += "\\\ "
            return output_string

        except Exception as e:
            print(f"Issue arose for folder {folder}, class: {class_name}.")
            print(traceback.format_exc())

    def plot_volume(self, study_folders, study_names):

        sns.set_style('ticks')
        save_fname = str(Path(self.output_folder, f'volume_scatter_{study_names[0]}.eps'))
        font = {'family': 'normal',
                'size': 10} #'weight': 'bold',

        matplotlib.rc('font', **font)
        # fig = plt.figure()
        fig, axs = plt.subplots(1, len(study_folders), sharex=True, sharey=True, figsize=(10, 3))
        plt.xscale('symlog')
        plt.yscale('symlog')

        for i, (sf, name) in enumerate(zip(study_folders, study_names)):
            if i == 0:
                ref_data = self.__read_and_threshold_results(sf, self.class_names[0])
                x_axis = 'GT volume (ml)'
                plot_data = ref_data
            else:
                data = self.__read_and_threshold_results(sf, self.class_names[0])
                x_axis = f'{name} volume (ml)'
                data.rename({'Detection volume (ml)': x_axis}, axis=1, inplace=True)
                plot_data = pd.merge(ref_data, data, on='Patient', how='left')

            corr_mat = plot_data[['Detection volume (ml)', x_axis]].corr()
            print(corr_mat)
            # sns.scatterplot(data=plot_data, x=x_axis, y='Detection volume (ml)', ax=axs[i])
            # axs[i].plot(axs[i].get_xlim(), axs[i].get_ylim(), ls="--", c=".1")
            p = sns.regplot(data=plot_data, x=x_axis, y='Detection volume (ml)', ax=axs[i])
            slope, intercept, r, p, sterr = scipy.stats.linregress(x=p.get_lines()[0].get_xdata(),
                                                                   y=p.get_lines()[0].get_ydata())
            print(f"Slope {slope}, intercept {intercept}")
            axs[i].text(3.1, -0.2, f'Slope = {slope:.3f}\nCorr.  = {corr_mat.values[0, 1]:.3f}', weight='bold')
            # ax.title('True vs predicted postop volume (symlog scale)')
            if i == 0:
                axs[i].set(ylabel='Predicted postop volume (ml)')
            else:
                axs[i].set(ylabel='')
            axs[i].set(xlabel=x_axis)
        # fig.legend(study_names)
        plt.tight_layout()
        plt.savefig(save_fname, format='eps')

    def __read_and_threshold_results(self, study_folder, class_name, category='All'):
        results_filename = os.path.join(study_folder, class_name + '_dice_scores.csv')
        results = pd.read_csv(results_filename)
        classes_optimal = self.__retrieve_optimum_values(study_folder, class_name)
        best_threshold = classes_optimal[class_name]['All'][1]

        results.replace('inf', np.nan, inplace=True)
        optimal_results = results.loc[results['Threshold'] == best_threshold]
        return optimal_results

    def __retrieve_optimum_values(self, study_folder, class_name: str):
        classes_optimal = {}
        study_filename = os.path.join(study_folder, class_name + '_optimal_dice_study.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this. Missing optimal_dice_study file.')

        optimal_overlap, optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)
        classes_optimal[class_name] = {}
        classes_optimal[class_name]['All'] = [optimal_overlap, optimal_threshold]

        study_filename = os.path.join(study_folder, class_name + '_optimal_dice_study_tp.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this. Missing optimal_dice_study_tp file.')

        optimal_overlap, optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)
        classes_optimal[class_name]['True Positive'] = [optimal_overlap, optimal_threshold]
        return classes_optimal