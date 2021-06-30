# read case 3 images (non aligned)
# align the reference image
# calc err: inspected - aligned_reference
# apply the model: p_defect(err[i, j]) / (p_defect(err[i, j]) + p_non_defect(err[i, j]))
# show probability map + binary map

import cv2
from load_data import load_images_pairs
# choose one of the 3 following alignment methods:
# from align_by_phase_correlate import AlignByPhaseCorrelate as AlignAlgo
# from align_by_convolutional_filter import AlignByConvolutionalFilter as AlignAlgo
from align_by_ECC import AlignByECC as AlignAlgo


class Inference:

    @staticmethod
    def align_image():
        images = load_images_pairs()  # the images we want to use in test-time

        # compute aligned reference images per each case:
        aa = AlignAlgo()
        aligned_images = aa.run_all(images=images)

        for i, case in enumerate(images):
            inspected = aligned_images[case]['inspected']
            shifted_reference_image = aligned_images[case]['shifted_reference_image']
            err = inspected - shifted_reference_image



inf = Inference()
inf.align_image()

