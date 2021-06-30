from typing import Tuple
import numpy as np
import cv2
from align_images_common import AlignImagesCommon


class AlignByConvolutionalFilter(AlignImagesCommon):

    def __init__(self):
        super(AlignByConvolutionalFilter, self).__init__()
        self.gaussian_blur_filter_size = 101
        self.gaussian_blur_filter_stdev = 5
        self.k_size = (51, 51)  # This will let us calc relative translation of up to 25 pixels (per each direction).
        # It is not enough, but the matrix dimensionality (degrees of freedom) is already 2601 here, so a significant
        # increase of these numbers is not highly recommended. What we can do, in case the translation is higher, is to
        # iteratively execute the 'return_aligned_reference_image' method, first with 'significantly' blurred image
        # (gaussian kernel etc.), apply the estimated translation, then reduce gradually the blur (filter width), apply
        # the new estimated translation and so on till convergence (I did not implement it here).

    def get_translation_params(self, inspected: np.array, reference: np.array) -> Tuple[float, float]:
        """
        this function computes the translation shifts in x and y directions of the reference w.r.t. the inspected image.
        basically the idea is to learn a convolution filter of the some shape (an h.p.) that will
        represent the corresponding shift.
        for example, a shift of 2 pixels will be represented (if convergence is successful) by
        a matrix of zeros with a single nonzero element 2 pixels from the center.
        loss function used is MSE (to utilize its closed-form solution).
        :param inspected: inspected image
        :param reference: reference image
        :return: translation shifts in x and y directions
        """

        # show original inspected image:
        # cv2.imshow('inspected orig', inspected)

        inspected = cv2.GaussianBlur(src=inspected, ksize=(self.gaussian_blur_filter_size, self.gaussian_blur_filter_size), sigmaX=self.gaussian_blur_filter_stdev)
        reference = cv2.GaussianBlur(src=reference, ksize=(self.gaussian_blur_filter_size, self.gaussian_blur_filter_size), sigmaX=self.gaussian_blur_filter_stdev)

        # show blurred inspected image:
        # cv2.imshow('inspected blurred', inspected)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        src1 = np.float32(inspected)
        src2 = np.float32(reference)

        assert src1.shape == src2.shape
        assert (self.k_size[0] % 2 == 1) & (self.k_size[1] % 2 == 1)  # we need an odd-sized kernel

        num_strides = (src1.shape[1] - self.k_size[1] + 1) * (src1.shape[0] - self.k_size[0] + 1)  # number of 'passes' of the kernel over the image

        print('start building A')
        # construct A: it is the transition matrix from the flattened kernel to the flattened reference image
        A = np.zeros((num_strides, np.prod(self.k_size)), dtype=float)
        c = 0
        for i in range(self.k_size[0]):
            mask_r = np.arange(i, src1.shape[0] - self.k_size[0] + i + 1)
            for j in range(self.k_size[1]):
                mask_c = np.arange(j, src1.shape[1] - self.k_size[1] + j + 1)
                A[:, c] = src1[mask_r[:, None], mask_c[None, :]].reshape(-1)
                c += 1

        print('start building b')
        # construct b: it is the flattened reference image
        mask_r = np.arange(self.k_size[0] // 2, src2.shape[0] - self.k_size[0] // 2)
        mask_c = np.arange(self.k_size[1] // 2, src2.shape[1] - self.k_size[1] // 2)
        b = src2[mask_r[:, None], mask_c[None, :]].reshape(-1)

        print('start solve Ax = b')
        # solve x for Ax = b: (x is the flattened kernel)
        At_A = A.T @ A
        At_b = A.T @ b
        x = np.linalg.inv(At_A) @ At_b  # x can be solved by cholesky decomp of At_A + forward-backward substituion, or by conj-grad algo. did not implement it here.
        k = x.reshape(self.k_size, order='C')
        print('done solving')

        (max_i, max_j) = np.unravel_index(np.argmax(k), k.shape)
        center_i, center_j = self.k_size[0] // 2, self.k_size[1] // 2

        delta_i = max_i - center_i
        delta_j = max_j - center_j

        return float(delta_i), float(delta_j)


if __name__ == '__main__':
    from load_data import load_images_pairs
    images = load_images_pairs()
    acf = AlignByConvolutionalFilter()
    aligned_images = acf.run_all(images=images)
    for case in aligned_images:
        for img_type in aligned_images[case]:
            img = aligned_images[case][img_type]
            cv2.imshow(f'{case} {img_type}', img)
            cv2.waitKey(delay=2000)
            cv2.destroyAllWindows()

# comment:
# this method has a the benefits of:
# - convergence in a few iterations (each iteration with decreasing blurring)
# - ability to handle very high spatial frequencies (we need to find only one 'matching' pixel per each stride)
# - robustness to different lightning conditions between both images (we only extract the argmax of the filter, the vale itself night be low, and we do not care for it).
# - mathematically interesting:) (if fast solving techniques are deployed)
# but the drawbacks are:
# - each iteration is computationally heavy
# - sensitivity to hp's selection (blurring params + kernel size)
# Currently it needs further tuning and the expansion to the iterative execution to properly work.
# For the time being, it is inferior to the other approached in this exercise.
