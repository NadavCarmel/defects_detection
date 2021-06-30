# read case 3 images (non aligned)
# align the reference image
# calc err: inspected - aligned_reference
# apply the model: p_defect(err[i, j]) / (p_defect(err[i, j]) + p_non_defect(err[i, j]))
# show probability map + binary map

import cv2
import numpy as np
import pickle
from load_data import load_images_pairs
# choose one of the 3 following alignment methods:
# from align_by_phase_correlate import AlignByPhaseCorrelate as AlignAlgo
# from align_by_convolutional_filter import AlignByConvolutionalFilter as AlignAlgo
from align_by_ECC import AlignByECC as AlignAlgo


class Inference:

    @staticmethod
    def load_model():
        with open('../results/model.pkl','rb') as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def align_images():
        images = load_images_pairs()  # the images we want to use in test-time

        # compute aligned reference images per each case:
        aa = AlignAlgo()
        aligned_images = aa.run_all(images=images)

        return aligned_images

    @staticmethod
    def calc_inspected_to_reference_diff(aligned_images):
        for i, case in enumerate(aligned_images):
            inspected = aligned_images[case]['inspected']
            shifted_reference_image = aligned_images[case]['shifted_reference_image']
            err = inspected - shifted_reference_image  # our diff array - will be used for the defects detection
            aligned_images[case]['err'] = err  # add 'err' per each case to 'aligned_images'
        return aligned_images

    @staticmethod
    def calc_prob(X, mu, sigma):
        Z = (X - mu) / sigma
        P = 1 / (2 * np.pi * sigma) * np.exp(-0.5 * (Z ** 2))  # todo: check later
        return P

    def predict(self, aligned_images, model):

        for case in aligned_images:
            err = aligned_images[case]['err']

            P_defects = self.calc_prob(err, mu=model['mu_err_defects_mean'], sigma=model['sigma_err_defects_mean'])
            P_non_defects = self.calc_prob(err, mu=model['mu_err_non_defects_mean'], sigma=model['sigma_err_non_defects_mean'])
            P_defects /= (P_defects + P_non_defects)

            prediction_mask = P_defects > 0.5

            aligned_images[case]['P_defects'] = P_defects
            aligned_images[case]['prediction_mask'] = prediction_mask

        return aligned_images

    def visualize(self, aligned_images):
        for case in aligned_images:
            cv2.imshow('shifted_reference_image', aligned_images[case]['shifted_reference_image'])
            cv2.imshow('inspected', aligned_images[case]['inspected'])
            cv2.imshow('err', aligned_images[case]['err'])
            cv2.imshow('P_defects', aligned_images[case]['P_defects'])
            cv2.imshow('prediction_mask', aligned_images[case]['prediction_mask'])
            cv2.waitKey()

    def run_all(self):
        model = self.load_model()
        aligned_images = self.align_images()  # load the images and apply the alignment algo
        aligned_images = self.calc_inspected_to_reference_diff(aligned_images)  # add the inspected-to-reference err image
        aligned_images = self.predict(aligned_images=aligned_images, model=model)
        self.visualize(aligned_images)
        print('done execution')


if __name__ == '__main__':
    inf = Inference()
    inf.run_all()

