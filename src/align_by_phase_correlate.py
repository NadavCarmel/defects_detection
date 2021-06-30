from typing import Tuple
import numpy as np
import cv2
from align_images_common import AlignImagesCommon


class AlignByPhaseCorrelate(AlignImagesCommon):

    def __init__(self):
        super(AlignByPhaseCorrelate, self).__init__()

    def get_translation_params(self, inspected: np.array, reference: np.array) -> Tuple[float, float]:
        """
        this function computes the translation shifts in x and y directions of the reference w.r.t. the inspected image.
        It is based on the FFT property: corr(x(t), y(t)) = IFFT(mult(FFT(x(t), conj(FFT(y(t)))))
        I assume there is no scale / rotation differences of the 2 inputs.
        :param inspected: inspected image
        :param reference: reference image
        :return: translation shifts in x and y directions
        """

        src1 = np.float32(inspected)
        src2 = np.float32(reference)

        # add padding (made no improvement on this case)
        # src1 = np.pad(array=src1, pad_width=((100, 100), (100, 100)))
        # src2 = np.pad(array=src2, pad_width=((100, 100), (100, 100)))
        window = np.ones_like(src1, dtype=np.float32)

        ret = cv2.phaseCorrelate(src1, src2, window=window)

        delta_x, delta_y = ret[0]

        return delta_x, delta_y


if __name__ == '__main__':
    from load_data import load_images_pairs
    images = load_images_pairs()
    apc = AlignByPhaseCorrelate()
    aligned_images = apc.run_all(images=images)
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
