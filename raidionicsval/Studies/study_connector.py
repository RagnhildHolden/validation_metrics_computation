from ..Utils.resources import SharedResources
from ..Studies.SegmentationStudy import SegmentationStudy
from ..Studies.PostopSegmentationStudy import PostopSegmentationStudy
from ..Studies.PostopInterraterStudy import PostopInterraterStudy
from ..Studies.ComparePostopStudies import ComparePostopStudy
from ..Studies.PostopAlexStudy import PostopAlexStudy


class StudyConnector:
    """
    Instantiate the proper study class corresponding to the user choice from the configuration file.
    """
    def __init__(self):
        self.perform_study = SharedResources.getInstance().studies_task

    def run(self):
        if self.perform_study == 'segmentation':
            processor = SegmentationStudy()
            processor.run()

        if self.perform_study == 'postop_segmentation':
            processor = PostopSegmentationStudy()
            processor.run()

        if self.perform_study == 'postop_interrater':
            processor = PostopInterraterStudy()
            processor.run()

        if self.perform_study == 'compare_postop':
            processor = ComparePostopStudy()
            processor.run()

        if self.perform_study == 'postop_alex_study':
            processor = PostopAlexStudy()
            processor.run()
