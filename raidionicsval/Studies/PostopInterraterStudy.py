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
from ..Computation.dice_computation import compute_dice, compute_dice_uncertain
from tqdm import tqdm
from math import ceil


class PostopInterraterStudy():

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

        self.interrater_origin_folder = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/interrater_data')
        self.annotator_groups = ['nov', 'exp']
        self.annotators = [f"{annot}{i}" for i in range(1, 5) for annot in self.annotator_groups]
        self.consensus_annotations = ['consensus-nov', 'consensus-exp', 'consensus-all-annotators',
                                      'strict-consensus-nov', 'strict-consensus-exp', 'strict-consensus-all-annotators',
                                      'nov-union', 'exp-union', 'all-union']
        self.all_annotators = self.annotators + self.consensus_annotations
        # self.all_annotators = ['nov-union', 'exp-union', 'all-union']
        self.interrater_output_folder = Path(
            '/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/InterraterValidation')
        self.annotators_output_folders = [Path(self.interrater_output_folder, annotator) for annotator in
                                          self.all_annotators]
        self.model_folder = Path(
            '/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation')


    def run(self):
        # self.reorganize_interrater_data()
        # self.write_interrater_cv_file()
        # self.run_validation_all_raters()
        # self.run_interrater_validation_all_experiments()
        # self.create_interrater_consensus_segmentations()
        # self.compute_dice_scores_multiple_references()
        self.interrater_study_summary()
        # exp_folder = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/exp9_center_around_tumor')
        # self.run_interrater_validation_one_experiment(exp_folder)


    def run_interrater_validation_all_experiments(self):
        experiments = ['exp1_patch_wise_150123', 'exp2_patch_wise_w_neg_190123', 'exp3_patch_wise_w_neg_scratch',
                       'exp4_pw_w_neg_pretrained_260124', 'exp5_patching_in_data_load_96x96x96',
                       'exp6_2levels_kernelsize7', 'exp7_kernelsize5', 'exp8_dilated_GT', 'exp9_center_around_tumor']
        base_experiment_folder = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation')
        for exp in experiments:
            print(f"Running validation on inter-rater dataset for experiment {exp}")
            self.run_interrater_validation_one_experiment(Path(base_experiment_folder, exp))

    def run_interrater_validation_one_experiment(self, exp_folder):
        exp_valid = ExperimentInterraterValidation(exp_folder)
        exp_valid.run()
        exp_valid.threshold_predictions()

    def run_validation_all_raters(self):
        # self.all_annotators = self.all_annotators[-6:]
        # self.annotators_output_folders = self.annotators_output_folders[-6:]
        for rater, rater_folder in zip(self.all_annotators, self.annotators_output_folders):
            print(f"Running validation for rater {rater}, output_folder = {rater_folder}")
            self.run_validation_one_rater(rater_folder)

    def run_validation_one_rater(self, rater_folder):
        rater_valid = RaterValidation(rater_folder)
        rater_valid.run()
        # rater_valid.threshold_predictions()

    def compute_dice_scores_multiple_references(self):
        ref_data_path = Path(
            "/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/InterraterValidation/nov1/predictions/0")
        patient_folders = [d.name for d in ref_data_path.iterdir() if d.is_dir()]

        metrics = ['Dice', 'volume_eval', 'residual_tumor_eval']
        columns = ['reference', 'pid', 'volume_ref', 'residual_tumor_ref', 'eval'] + metrics

        self.interrater_study_filepath = Path(self.output_folder, 'interrater_study.csv')
        reference_annotators = self.consensus_annotations[3:]
        model_annotators = ['exp9_center_around_tumor']
        eval_annotators = self.consensus_annotations[3:]
        if not self.interrater_study_filepath.exists():
            self.interrater_results_df = pd.DataFrame(columns=columns)
        else:
            self.interrater_results_df = pd.read_csv(self.interrater_study_filepath)

        for ind, patient_folder in enumerate(patient_folders):
            pid = patient_folder.split('_')[1]

            # Load reference segmentations
            references = self.__load_annotations(patient_folder, reference_annotators, model_annotators)

            print(f"Compute scores for patient pid {pid}")
            for ref_name, ref_ni in references.items():
                print(f"Reference: {ref_name}")

                # Get segmentations
                segmentations = self.__load_annotations(patient_folder, eval_annotators, model_annotators)

                # If any of them does not exist - compute reference volume + res tumor
                ref_volume, ref_residual_tumor = compute_volume_residual_tumor(ref_ni,
                                                                               threshold_segmentation(ref_ni, 0.6))
                ref_basic_info = [ref_name, pid, ref_volume, ref_residual_tumor]

                for eval_name, seg_ni in segmentations.items():
                    # Check for entries in results
                    eval_res = self.interrater_results_df.loc[(self.interrater_results_df['reference'] == ref_name) &
                                                              (self.interrater_results_df['pid'] == pid) &
                                                              (self.interrater_results_df[
                                                                   'eval'] == eval_name)]
                    if len(eval_res) != 0 and not np.isnan(np.sum(eval_res.values[5:])):
                        continue

                    if not seg_ni.shape == ref_ni.shape:
                        print(
                            f"Mismatch in shape between {eval_name} segmentation and GT for patient {pid}, skip")
                        continue
                    results = dice_computation(ref_ni, seg_ni, 0.6)
                    results_df = pd.DataFrame([ref_basic_info + [eval_name] + results], columns=columns)

                    self.interrater_results_df = self.interrater_results_df.append(results_df, ignore_index=True)
                    self.interrater_results_df.to_csv(self.interrater_study_filepath, index=False)

    def interrater_study_summary(self):
        tp_gt_pids = [4144, 3985, 3995, 3952, 4164, 3884, 4128, 4493, 4699, 4651]
        self.interrater_study_filepath = Path(self.output_folder, 'interrater_study.csv')
        results = pd.read_csv(self.interrater_study_filepath)
        results.replace('inf', 0, inplace=True)
        results.replace('', 0, inplace=True)
        results.replace(' ', 0, inplace=True)

        metrics = ['Dice', 'volume_eval', 'residual_tumor_eval']
        columns = ['reference', 'pid', 'volume_ref', 'residual_tumor_ref', 'eval'] + metrics

        # Compute metrics
        eval_metrics = ['Dice', 'Dice-P', 'Dice-P-GT']
        average_columns = ['reference', 'evaluator', 'Patient-wise recall', 'Patient-wise precision',
                           'Patient-wise specificity',
                           'Patient-wise F1', 'Accuracy', 'Balanced accuracy', 'Positive rate', 'Negative rate']
        for m in eval_metrics:
            average_columns.extend([m + '_mean', m + '_std'])

        unique_references = np.unique(results['reference'])
        unique_evaluators = np.unique(results['eval'])
        print(len(unique_evaluators), unique_evaluators)

        metrics_per_ref_evaluator = []
        for ref in unique_references:
            ref_results = results.loc[results['reference'] == ref]

            for evaluator in unique_evaluators:
                evaluator_average = []
                eval_results = ref_results.loc[ref_results['eval'] == evaluator]
                true_pos = len(
                    eval_results.loc[
                        (eval_results[f'residual_tumor_eval'] == 1) & (eval_results['residual_tumor_ref'] == 1)])
                false_pos = len(
                    eval_results.loc[
                        (eval_results[f'residual_tumor_eval'] == 1) & (eval_results['residual_tumor_ref'] == 0)])
                true_neg = len(
                    eval_results.loc[
                        (eval_results[f'residual_tumor_eval'] == 0) & (eval_results['residual_tumor_ref'] == 0)])
                false_neg = len(
                    eval_results.loc[
                        (eval_results[f'residual_tumor_eval'] == 0) & (eval_results['residual_tumor_ref'] == 1)])
                print(sum([true_pos, false_pos, true_neg, false_neg]))

                recall = 1 if (true_pos + false_neg) == 0 else true_pos / (true_pos + false_neg)
                precision = 1 if (true_pos + false_pos) == 0 else true_pos / (true_pos + false_pos)
                specificity = 1 if (true_neg + false_pos) == 0 else true_neg / (true_neg + false_pos)
                f1 = 2 * ((recall * precision) / (recall + precision))
                accuracy = (true_pos + true_neg) / sum([true_pos, false_pos, true_neg, false_neg])
                balanced_acc = (recall + specificity) / 2
                pos_rate = (true_pos + false_neg) / len(eval_results)
                neg_rate = (true_neg + false_pos) / len(eval_results)
                evaluator_average.extend([recall, precision, specificity, f1, accuracy, balanced_acc,
                                          pos_rate, neg_rate])

                for m in eval_metrics:
                    if '-P-GT':
                        positives = eval_results.loc[(eval_results['pid'].isin(tp_gt_pids))]
                        # avg = positives[f'{evaluator}_{m.split("-")[0]}'].astype('float32').mean()
                        # std = positives[f'{evaluator}_{m.split("-")[0]}'].astype('float32').std()
                        avg = positives[m.split('-')[0]].astype('float32').mean()
                        std = positives[m.split('-')[0]].astype('float32').std()
                        evaluator_average.extend([avg, std])
                    elif '-P' in m:
                        positives = eval_results.loc[(eval_results['residual_tumor_ref'] == 1)]
                        # avg = positives[f'{evaluator}_{m.split("-")[0]}'].astype('float32').mean()
                        # std = positives[f'{evaluator}_{m.split("-")[0]}'].astype('float32').std()
                        avg = positives[m.split('-')[0]].astype('float32').mean()
                        std = positives[m.split('-')[0]].astype('float32').std()
                        evaluator_average.extend([avg, std])

                    else:
                        # avg = ref_results[f'{evaluator}_{m.split("-")[0]}'].astype('float32').mean()
                        # std = ref_results[f'{evaluator}_{m.split("-")[0]}'].astype('float32').std()
                        avg = eval_results[m].astype('float32').mean()
                        std = eval_results[m].astype('float32').std()
                        evaluator_average.extend([avg, std])

                metrics_per_ref_evaluator.append([ref, evaluator] + evaluator_average)

        metrics_df = pd.DataFrame(metrics_per_ref_evaluator, columns=average_columns)
        output_filepath = Path(self.output_folder, 'interrater_study_average_metrics.csv')
        metrics_df.to_csv(output_filepath, index=False)

    def __load_annotations(self, patient_folder, annotators, model_annotators):
        references = {}
        for annot in annotators:
            annot_folder = Path(self.interrater_output_folder, annot, 'predictions/0', patient_folder)
            annotation_filepath = [f for f in annot_folder.iterdir() if f.is_file()]
            if len(annotation_filepath) == 0:
                print(f"No annotations for patient {patient_folder} annot {annot}")
            else:
                annotation_ni = nib.load(annotation_filepath[0])
                if len(annotation_ni.shape) == 4:
                    annotation_ni = nib.four_to_three(annotation_ni)[0]
            references[annot] = annotation_ni
        for annot in model_annotators:
            annot_folder = Path(self.model_folder, annot, 'external_test_predictions/0', patient_folder)
            annotation_filepath = [f for f in annot_folder.iterdir() if f.is_file()]
            if len(annotation_filepath) == 0:
                print(f"No annotations for patient {patient_folder} annot {annot}")
            else:
                annotation_ni = nib.load(annotation_filepath[0])
                if len(annotation_ni.shape) == 4:
                    annotation_ni = nib.four_to_three(annotation_ni)[0]
            references[annot] = annotation_ni

        return references

    def create_interrater_consensus_segmentations(self):
        ref_data_path = Path("/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/InterraterValidation/nov1/predictions/0")
        patient_folders = [d.name for d in ref_data_path.iterdir() if d.is_dir()]

        annotator_groups = ['nov', 'exp']
        annotators = [[f"{annot}{i}" for i in range(1, 5)] for annot in annotator_groups]

        union_annotation_folders = ['nov-union', 'exp-union', 'all-union']
        output_folders = [Path(self.interrater_output_folder, d) for d in union_annotation_folders]
        for d in output_folders:
            d.mkdir(exist_ok=True)

        for ind, patient_folder in enumerate(patient_folders):
            all_loaded_annotations = []
            for j, group in enumerate(annotator_groups):

                group_loaded_annotations = []
                for annotator in annotators[j]:
                    annot_folder = Path(self.interrater_output_folder, annotator, 'predictions/0', patient_folder)
                    annotation_filepath = [f for f in annot_folder.iterdir() if f.is_file()]
                    if len(annotation_filepath) == 0:
                        print(f"No annotations for patient {patient_folder} annot {annotator}, creating empty mask")
                    else:
                        annotation_ni = nib.load(annotation_filepath[0])
                        if len(annotation_ni.shape) == 4:
                            annotation_ni = nib.four_to_three(annotation_ni)[0]
                    group_loaded_annotations.append(annotation_ni.get_data())
                    all_loaded_annotations.append(annotation_ni.get_data())

                # consensus_group = np.mean(np.array(group_loaded_annotations), axis=0)
                # consensus_group[consensus_group > 0.5] = 1
                # consensus_group[consensus_group <= 0.5] = 0
                # print(consensus_group.shape, np.min(consensus_group), np.max(consensus_group))
                union_group = np.max(np.array(group_loaded_annotations), axis=0)
                print(f"Min {np.min(union_group)}, max {np.max(union_group)}, shape {np.shape(union_group)}")
                group_output_folder = Path(self.interrater_output_folder, f'{group}-union', 'predictions/0', patient_folder)
                group_output_folder.mkdir(exist_ok=True, parents=True)
                output_filepath = Path(group_output_folder, annotation_filepath[0].name)
                nib.save(nib.Nifti1Image(union_group, affine=annotation_ni.affine), output_filepath)

            union_all = np.max(np.array(all_loaded_annotations), axis=0)
            print(f"Min {np.min(union_group)}, max {np.max(union_group)}, shape {np.shape(union_group)}")
            all_output_folder = Path(self.interrater_output_folder, f'all-union', 'predictions/0', patient_folder)
            all_output_folder.mkdir(exist_ok=True, parents=True)
            output_filepath = Path(all_output_folder, annotation_filepath[0].name)
            nib.save(nib.Nifti1Image(union_all, affine=annotation_ni.affine), output_filepath)
            # print(consensus_all.shape, np.min(consensus_all), np.max(consensus_all))

    def reorganize_interrater_data(self):
        interrater_data_path = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/interrater_data')
        interrater_glioblastoma_data_path = Path(interrater_data_path, 'Glioblastoma')
        patient_folders = [d for d in interrater_glioblastoma_data_path.iterdir() if d.is_dir()]
        patient_op_ids = [int(d.name.split('-')[1][4:]) for d in patient_folders]

        patient_id_mapping_filepath = Path(
            '/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/patient_id_mapping.csv')
        id_df = pd.read_csv(patient_id_mapping_filepath)
        vumc_df = id_df[(id_df['Hospital'] == 'VUmc')]

        reference_experiment_dir = Path('/home/ragnhild/Data/Neuro/Studies/PostopSegmentation/NetworkValidation/exp1_patch_wise_150123')
        cross_validation_description_file = os.path.join(reference_experiment_dir, 'external_testset.txt')
        external_test_set, _ = get_fold_from_file(filename=cross_validation_description_file, fold_number=0)

        self.interrater_test_set = []

        for ind, (op_id, patient_folder) in enumerate(zip(patient_op_ids, patient_folders)):
            patient = vumc_df[vumc_df['OP.ID'] == op_id]
            pid = patient['DB_ID'].values[0]
            db_index = patient['DB_index'].values[0]
            ext_test_set_name = [fname for fname in external_test_set if f'{db_index}_{pid}' in fname][0]
            self.interrater_test_set.append(ext_test_set_name)

            loaded_annotations = []

            # Load T1c image to check shape
            original_annotation_folder = Path(patient_folder, 'ses-postop', 'anat')
            image_filepath = [f for f in original_annotation_folder.iterdir() if 'NativeT1c' in f.name and 'dseg' not in f.name][
                0]
            # image_ni = nib.load(image_filepath)
            fname = '_'.join(ext_test_set_name.split('_')[1:-1]) + '-pred_tumor.nii.gz'
            ref_annotation_filepath = Path(reference_experiment_dir, 'external_test_predictions', '0', f'{db_index}_{pid}',
                                           fname)
            ref_annotation = nib.load(ref_annotation_filepath)
            all_annotations = [f for f in original_annotation_folder.iterdir() if
                               'label' in f.name and 'TMRenh_dseg' in f.name and
                               'NativeT1c' in f.name]

            for annot, annotator_folder in zip(self.all_annotators, self.annotators_output_folders):
                output_dir = Path(annotator_folder, 'predictions', '0', f'{db_index}_{pid}')
                output_dir.mkdir(exist_ok=True, parents=True)

                original_annotation_filepath = [a for a in all_annotations if 'label-' + annot in a.name]
                copy_annotation_filepath = Path(output_dir, fname)

                if copy_annotation_filepath.exists() and not self.overwrite:
                    print(f"{copy_annotation_filepath.name} exists, skip")
                    continue

                if len(original_annotation_filepath):
                    # Copy the annotation to destination
                    original_annotation_filepath = original_annotation_filepath[0]
                    annotation = nib.load(original_annotation_filepath)
                    if not annotation.shape == ref_annotation.shape:
                        print(f"Error, annotation shape for original annotation {original_annotation_filepath.name} \\ "
                              f"does not match reference annotation, {annotation.shape} != {ref_annotation.shape}")
                    else:
                        print(f"Copy annotation from {original_annotation_filepath.name} to {copy_annotation_filepath.name}")
                        nib.save(annotation, copy_annotation_filepath)
                else:
                    print("Annotation does not exist, create empty annotation")
                    annotation = np.zeros(ref_annotation.shape)
                    annotation_ni = nib.Nifti1Image(annotation, affine=ref_annotation.affine)
                    nib.save(annotation_ni, copy_annotation_filepath)

    def write_interrater_cv_file(self):
        for annotator_folder in self.annotators_output_folders:
            output_file = Path(annotator_folder, 'interrater_testset.txt')
            f = open(output_file, 'w')
            f.write(' '.join([str(x) for x in self.interrater_test_set]) + '\n')
            f.write(' '.join([str(x) for x in self.interrater_test_set]) + '\n')
            f.close()

        output_file = Path(self.interrater_output_folder, 'interrater_testset.txt')
        f = open(output_file, 'w')
        f.write(' '.join([str(x) for x in self.interrater_test_set]) + '\n')
        f.write(' '.join([str(x) for x in self.interrater_test_set]) + '\n')
        f.close()

class RaterValidation(ModelValidation):
    """
    Compute performances metrics after k-fold cross-validation from sets of inference.
    The results will be stored inside a Validation sub-folder placed within the provided destination directory.
    """
    def __init__(self, input_folder):
        self.data_root = SharedResources.getInstance().data_root
        self.input_folder = input_folder

        print(f"Running model validation for experiment {Path(self.input_folder).name}")
        self.prediction_folder = os.path.join(self.input_folder, 'predictions')
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

    def threshold_predictions(self):
        class_name = SharedResources.getInstance().validation_class_names[0]
        self.__retrieve_optimum_values(class_name)
        opt_thresh = self.classes_optimal[class_name]['All'][1]
        prediction_folder = Path(self.prediction_folder, '0')
        output_folder = Path(self.input_folder, f'{Path(self.prediction_folder).name}_thresholded', '0')
        output_folder.mkdir(exist_ok=True, parents=True)
        data_list, _ = get_fold_from_file(filename=self.cross_validation_description_file, fold_number=0)
        for i, patient in enumerate(tqdm(data_list)):
            pid = patient.split('_')[1]
            folder_id = "_".join(patient.split("_")[:2])
            patient_folder = Path(prediction_folder, folder_id)
            if patient_folder.exists():
                pred_files = [f for f in patient_folder.iterdir()]
                for p in pred_files:
                    prediction_ni = nib.load(p)
                    thresh_pred = threshold_segmentation(prediction_ni, opt_thresh)
                    patient_output_folder = Path(output_folder, folder_id)
                    patient_output_folder.mkdir(exist_ok=True)
                    output_fname = Path(patient_output_folder, p.name)
                    nib.save(nib.Nifti1Image(thresh_pred, prediction_ni.affine), output_fname)

    def __retrieve_optimum_values(self, class_name: str):
        self.classes_optimal = {}
        study_filename = os.path.join(self.input_folder, 'InterraterStudy', class_name + '_optimal_dice_study.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this. Missing optimal_dice_study file.')

        optimal_overlap, optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)
        self.classes_optimal[class_name] = {}
        self.classes_optimal[class_name]['All'] = [optimal_overlap, optimal_threshold]


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

    def threshold_predictions(self):
        class_name = SharedResources.getInstance().validation_class_names[0]
        self.__retrieve_optimum_values(class_name)
        opt_thresh = self.classes_optimal[class_name]['All'][1]
        prediction_folder = Path(self.prediction_folder, '0')
        output_folder = Path(self.input_folder, f'{Path(self.prediction_folder).name}_thresholded', '0')
        output_folder.mkdir(exist_ok=True, parents=True)
        data_list, _ = get_fold_from_file(filename=self.cross_validation_description_file, fold_number=0)
        for i, patient in enumerate(tqdm(data_list)):
            pid = patient.split('_')[1]
            folder_id = "_".join(patient.split("_")[:2])
            patient_folder = Path(prediction_folder, folder_id)
            if patient_folder.exists():
                pred_files = [f for f in patient_folder.iterdir()]
                for p in pred_files:
                    prediction_ni = nib.load(p)
                    thresh_pred = threshold_segmentation(prediction_ni, opt_thresh)
                    patient_output_folder = Path(output_folder, folder_id)
                    patient_output_folder.mkdir(exist_ok=True)
                    output_fname = Path(patient_output_folder, p.name)
                    nib.save(nib.Nifti1Image(thresh_pred, prediction_ni.affine), output_fname)

    def __retrieve_optimum_values(self, class_name: str):
        self.classes_optimal = {}
        study_filename = os.path.join(self.input_folder, 'Validation', class_name + '_optimal_dice_study.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this. Missing optimal_dice_study file.')

        optimal_overlap, optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)
        self.classes_optimal[class_name] = {}
        self.classes_optimal[class_name]['All'] = [optimal_overlap, optimal_threshold]

        study_filename = os.path.join(self.input_folder, 'Validation', class_name + '_optimal_dice_study_tp.csv')
        if not os.path.exists(study_filename):
            raise ValueError('The validation task must be run prior to this. Missing optimal_dice_study_tp file.')

        optimal_overlap, optimal_threshold = reload_optimal_validation_parameters(study_filename=study_filename)
        self.classes_optimal[class_name]['True Positive'] = [optimal_overlap, optimal_threshold]

def dice_computation(reference_ni, detection_ni, t=0.5):
    reference = threshold_segmentation(reference_ni, t)
    detection = threshold_segmentation(detection_ni, t)

    dice = compute_dice(reference, detection)
    # jaccard = dice / (2-dice)
    volume_seg_ml, res_tumor = compute_volume_residual_tumor(detection_ni, detection)
    return [dice, volume_seg_ml, res_tumor] # jaccard


def threshold_segmentation(segmentation_ni, t):
    # segmentation = deepcopy(segmentation_ni.get_data())
    segmentation = np.zeros(segmentation_ni.shape, dtype=np.uint8)
    # segmentation[segmentation <= t] = 0
    segmentation[segmentation_ni.get_data() > t] = 1
    # return segmentation.astype('uint8')
    return segmentation.astype('uint8')

def compute_volume_residual_tumor(segmentation_ni, segmentation):
    voxel_size = np.prod(segmentation_ni.header.get_zooms()[0:3])
    volume_pixels = np.count_nonzero(segmentation != 0)  # Might be more than one label, but not considering it yet
    volume_mmcube = voxel_size * volume_pixels
    volume_seg_ml = volume_mmcube * 1e-3

    res_tumor = 1 if volume_seg_ml > 0.175 else 0

    return volume_seg_ml, res_tumor
