import multiprocessing
import itertools

import time
import pandas as pd
from math import ceil
import os

from tqdm import tqdm

from ..Computation.dice_computation import separate_dice_computation
from ..Validation.instance_segmentation_validation import *
from ..Utils.resources import SharedResources
from ..Utils.PatientMetricsStructure import PatientMetrics
from ..Utils.io_converters import get_fold_from_file, reload_optimal_validation_parameters
from ..Validation.validation_utilities import best_segmentation_probability_threshold_analysis, compute_fold_average
from ..Validation.extra_metrics_computation import compute_patient_extra_metrics
from ..Validation.kfold_model_validation import ModelValidation


class ExternalModelValidation(ModelValidation):
    """
    Compute performances metrics after k-fold cross-validation from sets of inference.
    The results will be stored inside a Validation sub-folder placed within the provided destination directory.
    """
    def __init__(self):
        self.data_root = SharedResources.getInstance().data_root
        self.input_folder = SharedResources.getInstance().validation_input_folder
        self.prediction_folder = os.path.join(self.input_folder, 'external_test_predictions')
        base_output_folder = SharedResources.getInstance().validation_output_folder

        if base_output_folder is not None and base_output_folder != "":
            self.output_folder = os.path.join(base_output_folder, 'ExternalValidation')
        else:
            self.output_folder = os.path.join(self.input_folder, 'ExternalValidation')
        os.makedirs(self.output_folder, exist_ok=True)

        self.fold_number = SharedResources.getInstance().validation_nb_folds
        self.split_way = SharedResources.getInstance().validation_split_way
        self.metric_names = []
        self.metric_names.extend(SharedResources.getInstance().validation_metric_names)
        self.detection_overlap_thresholds = SharedResources.getInstance().validation_detection_overlap_thresholds
        print("Detection overlap: ", self.detection_overlap_thresholds)
        self.gt_files_suffix = SharedResources.getInstance().validation_gt_files_suffix
        self.prediction_files_suffix = SharedResources.getInstance().validation_prediction_files_suffix
        self.patients_metrics = {}

    def run(self):
        self.__compute_metrics()
        class_optimal = best_segmentation_probability_threshold_analysis(self.output_folder,
                                                                         detection_overlap_thresholds=self.detection_overlap_thresholds)
        if len(SharedResources.getInstance().validation_metric_names) != 0:
            self.__compute_extra_metrics(class_optimal=class_optimal)
        compute_fold_average(self.input_folder, class_optimal=class_optimal, metrics=self.metric_names,
                             true_positive_state=False)
        compute_fold_average(self.input_folder, class_optimal=class_optimal, metrics=self.metric_names,
                             true_positive_state=True)
