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
from ..Validation.kfold_model_validation import ModelValidation
from ..Studies.PostopInterraterStudy import ExperimentInterraterValidation


class PostopAlexStudy():

    def __init__(self):
        self.input_folder = SharedResources.getInstance().studies_input_folder
        self.output_folder = SharedResources.getInstance().studies_output_folder
        self.metric_names = []
        self.extra_patient_parameters = None
        if os.path.exists(SharedResources.getInstance().studies_extra_parameters_filename):
            self.extra_patient_parameters = pd.read_csv(SharedResources.getInstance().studies_extra_parameters_filename)
            # Patient unique ID might include characters
            self.extra_patient_parameters['Patient'] = self.extra_patient_parameters.Patient.astype(str)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        self.overwrite = False

        self.study_origin_folder = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/old_experiments/TRD_experiments')
        self.experiments = ['run2_exp2_T1c_T1w']
        self.study_output_folder = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/old_experiments/TRD_experiments')
        self.experiment_output_folders = [Path(self.study_origin_folder, exp) for exp in
                                          self.experiments]

    def run(self):
        self.run_validation_all_experiments()

    def run_validation_all_experiments(self):
        for exp in self.experiments:
            experiment_folder = Path(self.study_origin_folder, exp)

            # print(f"Reorganize predictions exp {exp}")
            # self.reorganize_predictions(experiment_folder)

            print(f"Running external test validation for experiment {exp}")
            self.run_external_validation_one_experiment(experiment_folder)

            print(f"Running validation on inter-rater dataset for experiment {exp}")
            self.run_interrater_validation_one_experiment(experiment_folder)

    def run_external_validation_one_experiment(self, exp_folder):
        exp_valid = ExperimentValidation(exp_folder)
        exp_valid.run()
    def run_interrater_validation_one_experiment(self, exp_folder):
        exp_valid = ExperimentInterraterValidation(exp_folder)
        exp_valid.run()

    def reorganize_predictions(self, experiment_folder):
        prediction_folder = Path(experiment_folder, 'test_predictions')
        new_prediction_folder = Path(experiment_folder, 'external_test_predictions')
        if prediction_folder.exists():
            os.rename(prediction_folder, new_prediction_folder)

        new_prediction_folder = Path(new_prediction_folder, '0')
        sub_folders = [d for d in new_prediction_folder.iterdir() if d.is_dir() and 'HGG' in d.name]
        pids = [d.name.split('_')[1] for d in sub_folders]

        ref_exp_folder = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/exp8_dilated_GT/external_test_predictions/0')
        ref_sub_folders = [d for d in ref_exp_folder.iterdir() if d.is_dir()]
        ref_pids = [d.name.split('_')[1] for d in ref_sub_folders]
        for sf, pid in zip(sub_folders, pids):
            ref_pid_index = [i for i in range(len(ref_pids)) if ref_pids[i] == pid][0]
            ref_sf = ref_sub_folders[ref_pid_index]
            ref_fname = [f for f in ref_sf.iterdir() if f.is_file() and f.name.endswith('.nii.gz')][0].name

            # Rename file
            pred_file = [f for f in sf.iterdir() if f.is_file() and f.name.endswith('.nii.gz')][0]
            os.rename(pred_file, Path(sf, ref_fname))

            # Rename dir
            os.rename(sf, Path(sf.parent, ref_sf.name))

        # Copy external testset file and inter-rater file
        ext_testset_file = Path(ref_exp_folder.parent.parent, 'external_testset.txt')
        copyfile(ext_testset_file, Path(new_prediction_folder.parent.parent, ext_testset_file.name))
        interrater_test_file = Path(ref_exp_folder.parent.parent, 'interrater_testset.txt')
        copyfile(interrater_test_file, Path(new_prediction_folder.parent.parent, interrater_test_file.name))


class ExperimentValidation(ModelValidation):
    """
    Compute performances metrics after k-fold cross-validation from sets of inference.
    The results will be stored inside a Validation sub-folder placed within the provided destination directory.
    """
    def __init__(self, input_folder):
        self.data_root = SharedResources.getInstance().data_root
        self.input_folder = input_folder

        print(f"Running model validation for experiment {Path(self.input_folder).name}")
        self.prediction_folder = os.path.join(self.input_folder, 'external_test_predictions')
        self.cross_validation_description_file = os.path.join(self.input_folder, 'external_testset.txt')
        val_foldername = 'ExternalValidation'
        self.split_way = 'two-way'
        self.output_folder = os.path.join(self.input_folder, val_foldername)
        os.makedirs(self.output_folder, exist_ok=True)

        self.fold_number = 1
        self.metric_names = []
        self.metric_names.extend(SharedResources.getInstance().validation_metric_names)
        self.detection_overlap_thresholds = SharedResources.getInstance().validation_detection_overlap_thresholds
        print("Detection overlap: ", self.detection_overlap_thresholds)
        self.gt_files_suffix = SharedResources.getInstance().validation_gt_files_suffix
        self.prediction_files_suffix = SharedResources.getInstance().validation_prediction_files_suffix
        self.patients_metrics = {}


class ExperimentInterraterValidation(ModelValidation):
    """
    Compute performances metrics after k-fold cross-validation from sets of inference.
    The results will be stored inside a Validation sub-folder placed within the provided destination directory.
    """
    def __init__(self, input_folder):
        self.data_root = SharedResources.getInstance().data_root
        self.input_folder = input_folder

        print(f"Running model validation for experiment {Path(self.input_folder).name}")
        self.prediction_folder = os.path.join(self.input_folder, 'external_test_predictions')
        self.cross_validation_description_file = os.path.join(self.input_folder, 'interrater_testset.txt')
        val_foldername = 'InterraterStudy'
        self.split_way = 'two-way'
        self.output_folder = os.path.join(self.input_folder, val_foldername)
        os.makedirs(self.output_folder, exist_ok=True)

        self.fold_number = 1
        self.metric_names = []
        self.metric_names.extend(SharedResources.getInstance().validation_metric_names)
        self.detection_overlap_thresholds = SharedResources.getInstance().validation_detection_overlap_thresholds
        print("Detection overlap: ", self.detection_overlap_thresholds)
        self.gt_files_suffix = SharedResources.getInstance().validation_gt_files_suffix
        self.prediction_files_suffix = SharedResources.getInstance().validation_prediction_files_suffix
        self.patients_metrics = {}
