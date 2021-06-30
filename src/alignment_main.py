import pickle
from load_data import load_images_pairs
# choose one of the 3 following alignment methods:
# from align_by_phase_correlate import AlignByPhaseCorrelate as AlignAlgo
# from align_by_convolutional_filter import AlignByConvolutionalFilter as AlignAlgo
from align_by_ECC import AlignByECC as AlignAlgo

# compute aligned reference images per each case:
aa = AlignAlgo()
images = load_images_pairs()
aligned_images = aa.run_all(images=images)

# save to results dir:
with open('../results/aligned_images.pkl', 'wb') as f:
    pickle.dump(aligned_images, f, pickle.HIGHEST_PROTOCOL)

print('\naligned images saved to results dir')

