import pickle
import numpy as np
from load_data import load_labels


class RunningMeanStats:

    def __init__(self):
        self.running_mean_mu = 0
        self.running_mean_sigma = 0
        self.counter = 0

    def update_mean_stats(self, x):
        self.running_mean_mu = self.running_mean_mu * self.counter + np.mean(x) * len(x)
        self.running_mean_sigma = self.running_mean_sigma * self.counter + np.std(x) * len(x)
        self.counter += len(x)
        self.running_mean_mu /= self.counter
        self.running_mean_sigma /= self.counter


class EstimateDefectsStats:

    @staticmethod
    def load_data():

        # load images (after alignment):
        with open('../results/aligned_images.pkl', 'rb') as f:
            aligned_images = pickle.load(f)

        # load labels:
        labels = load_labels()

        return aligned_images, labels

    def calc_inspected_to_reference_diff_stats(self, aligned_images, labels):
        # The running-means of the error statistics.
        # These are the only 4 'learnable' params in this task.
        running_mean_defects_stats = RunningMeanStats()
        running_mean_non_defects_stats = RunningMeanStats()

        for i, case in enumerate(labels):
            inspected = aligned_images[case]['inspected']
            shifted_reference_image = aligned_images[case]['shifted_reference_image']
            err = inspected - shifted_reference_image  # our diff array - will be used for the defects detection

            # visualize images:
            import cv2
            cv2.imshow('inspected', inspected)
            cv2.imshow('shifted_reference_image', shifted_reference_image)
            cv2.imshow('err', err)
            cv2.waitKey(delay=1000)
            cv2.destroyAllWindows()

            defects_mask = np.zeros_like(err, dtype=bool)
            for defect_idx in labels[case]:
                defects_mask[defect_idx] = True

            running_mean_defects_stats.update_mean_stats(err[defects_mask])
            running_mean_non_defects_stats.update_mean_stats(err[~defects_mask])

        model = {'mu_err_defects_mean': running_mean_defects_stats.running_mean_mu,
                 'sigma_err_defects_mean': running_mean_defects_stats.running_mean_sigma,
                 'mu_err_non_defects_mean': running_mean_non_defects_stats.running_mean_mu,
                 'sigma_err_non_defects_mean': running_mean_non_defects_stats.running_mean_sigma}

        return model

    def run_all(self):
        aligned_images, labels = self.load_data()
        model = self.calc_inspected_to_reference_diff_stats(aligned_images=aligned_images,
                                                            labels=labels)

        # save model:
        with open('../results/model.pkl', 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

        print('\nsuccessfully estimated and saved model.')
        return model


if __name__ == '__main__':
    eds = EstimateDefectsStats()
    model = eds.run_all()


