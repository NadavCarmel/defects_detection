Abstract
--------
This python project is designed to detect defects in images.
input: (inspected, reference) images pair
output: binary image ('prediction_mask') of the inspected image

Usage
-----
To execute the code:
1. Validate that the interpreter has all the dependencies from ./requirements.txt (it is a very slim file), or use:
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   $ pip install -r requirements.txt
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
2. Execute the alignment step by running ./src/alignment_main.py
   There will be 3 algorithmic options to choose from (at the 'imports' rows), so select the desired one and comment on the others.
   The alignment is of the reference image (we want to stay at the inspected image coordinate system).
   Aligned images will be saved as a dictionary to ./results/aligned_images.pkl, and will be used in later steps.
3. Calculate the difference array ('err') between each pair (inspected, shifted_reference) by running ./src/estimate_defects_model.py.
   This matrix will be used to locate the defects.
   Also, this code estimates a 'model': the mean and variance of the defected pixels, and of the normal pixels, and averages that over all images pairs.
   It saves those statistics to ./results/model.pkl.
4. At inference, run ./src/inference.py to predict the defects in the images. 
   It will load the 'model' and the images and compute, per each pixel in each 'err' array, whether it is more likely it was drawn from a defected distribution or from a non-defected distribution.
   It will return the binary defects map.


