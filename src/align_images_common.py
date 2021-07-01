from typing import Tuple, Dict
import numpy as np
import cv2
from load_data import load_images_pairs


class AlignImagesCommon:

    def get_translation_params(self, inspected: np.array, reference: np.array) -> Tuple[float, float]:
        pass

    def return_aligned_reference_image(self, inspected: np.array, reference: np.array) -> np.array:
        delta_x, delta_y = self.get_translation_params(inspected=inspected, reference=reference)
        shifted_reference_image = self.shift_image(image=reference, delta_x=delta_x, delta_y=delta_y)
        return shifted_reference_image

    @staticmethod
    def shift_image(image: np.array, delta_x: float, delta_y: float) -> np.array:
        M = np.array([[1, 0, -delta_x],
                      [0, 1, -delta_y]],
                     dtype=float)
        return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    def run_all(self, images) -> Dict[str, Dict[str, np.array]]:

        for case in images:
            print(f'\nworking on {case} alignment...')
            pair = images[case]
            shifted_reference_image = self.return_aligned_reference_image(inspected=pair['inspected'],
                                                                          reference=pair['reference'])
            images[case]['shifted_reference_image'] = shifted_reference_image

        return images

