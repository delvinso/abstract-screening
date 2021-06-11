# Refactored Experiment Code for Systematic Abstract Review

This is a WIP and works to streamline the experiment process as the code contained in `src` is cumbersome and difficult to navigate due to the embedding and modeling process being dependent on one another.
In contrast, the code in this folder separates the embedding and modeling process. 


1. embed_all_datasets.sh (except SPECTER which can be done by hand through data_preprocessing/notebooks/...)
2. check_pickles.py
3. run_all_datasets_x2_cv.sh