from typing import Dict
import numpy as np
import cv2
from constants import IMAGES_PATH


def load_images_pairs() -> Dict[str, Dict[str, np.array]]:
    images = {
        # defected
        'case1': {
            'inspected': cv2.imread(IMAGES_PATH + 'defective_examples/case1_inspected_image.tif'),
            'reference': cv2.imread(IMAGES_PATH + 'defective_examples/case1_reference_image.tif')
        },
        'case2': {
            'inspected': cv2.imread(IMAGES_PATH + 'defective_examples/case2_inspected_image.tif'),
            'reference': cv2.imread(IMAGES_PATH + 'defective_examples/case2_reference_image.tif')
        },
        # non defected
        'case3': {
            'inspected': cv2.imread(IMAGES_PATH + 'non_defective_examples/case3_inspected_image.tif'),
            'reference': cv2.imread(IMAGES_PATH + 'non_defective_examples/case3_reference_image.tif')
        }
    }

    for case in images:
        for img_type in images[case]:
            # compress channels when no 'real' RGB exists (I validated no image has RGB)
            channels_var = images[case][img_type].var(axis=2)
            max_var = np.max(channels_var)
            print(f'{case} {img_type} max_var: {max_var}')
            if max_var == 0:
                # no channels information => convert to grayscale
                images[case][img_type] = cv2.cvtColor(images[case][img_type], cv2.COLOR_BGR2GRAY)

    return images


def load_labels():
    # load labels:
    labels = {
        'case1': [(149, 334),
                  (82, 245),
                  (97, 82)],

        'case2': [(344, 265),
                  (105, 108),
                  (80, 262)]
    }
    return labels

