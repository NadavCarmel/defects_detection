from typing import Tuple
import numpy as np
import cv2
from align_images_common import AlignImagesCommon


class AlignByECC(AlignImagesCommon):

    def __init__(self):
        super(AlignByECC, self).__init__()
        self.number_of_iterations = 5000
        self.termination_eps = 1e-10
        self.gaussFiltSize = 151

    def get_translation_params(self, inspected: np.array, reference: np.array) -> Tuple[float, float]:
        """
        this function computes the translation shifts in x and y directions of the reference w.r.t. the inspected image.
        It is based on the minimizing the Enhanced Correlation Coefficient scheme (Georgios et al.)
        The idea is that the intensities o the images are normalized, thus the loss is insensitive to different light
        conditions between the images.
        :param inspected: inspected image
        :param reference: reference image
        :return: translation shifts in x and y directions
        """

        src1 = np.float32(inspected)
        src2 = np.float32(reference)

        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION

        # Define 2x3 matrix and initialize it to identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.number_of_iterations,
                    self.termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        (cc, warp_matrix) = cv2.findTransformECC(src1,
                                                 src2,
                                                 warp_matrix,
                                                 warp_mode,
                                                 criteria,
                                                 inputMask=None,
                                                 gaussFiltSize=self.gaussFiltSize)

        delta_x = warp_matrix[0, -1]
        delta_y = warp_matrix[1, -1]

        return delta_x, delta_y


if __name__ == '__main__':
    from load_data import load_images_pairs
    images = load_images_pairs()
    aec = AlignByECC()
    aligned_images = aec.run_all(images=images)
    for case in aligned_images:
        for img_type in aligned_images[case]:
            img = aligned_images[case][img_type]
            cv2.imshow(f'{case} {img_type}', img)
            cv2.waitKey(delay=2000)
            cv2.destroyAllWindows()

# comment:
# for case 1 the alignment seems very poor.
# for cases 2 & 3 it seems reasonably well.
# this is likely due to some high frequency components that are more significant in case 1.
# we could use some blurring step in advance (gaussian filter etc.), but I did try it.
