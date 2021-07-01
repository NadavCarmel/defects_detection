Abstract
--------
This python project is designed to detect defects in images.
input: (inspected, reference) images pair
output: binary image ('prediction_mask') of the inspected image

Usage
-----
To execute the code:
1. validate that the interpreter has all the dependencies from ./requirements.txt (it is a very slim file).
   or use:
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   $ pip install -r requirements.txt
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   
2. execute the alignment step by running ./src/alignment_main.py
   there will be 3 algorithmic options to choose from (at the 'imports' rows),so select the desired one and comment the others.
   the alignment is of the reference image (we want to stay at the inspected image coordinate system).
   aligned images will be saved as a dictionary to ./results/aligned_images.pkl, and will be used in later steps.
3. calculate the difference array ('err') between each pair (inspected, shifted_reference) by running ./src/estimate_defects_model.py.
   this matrix will be used to locate the defects.
   also, this code estimates a 'model': the mean and variance of the defected pixels, and of the normal pixels, and averages that over all images pairs.
   it saves those statistics to ./results/model.pkl.
4. at inference, run ./src/inference.py to predict the defects in the images. 
   it will load the 'model' and the images and compute, per each pixel in each 'err' array, weather it is more likely it was drawn from a defected distribution of from a non-defected distribution.
   it will return the binary defects map.


