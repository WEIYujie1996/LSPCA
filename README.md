# Supervised PCA

This code accompanies our work:

@article{ritchie2020supervised,   
  title={Supervised PCA: A Multiobjective Approach},   
  author={Ritchie, Alexander and Balzano, Laura and Kessler, Daniel and Sripada, Chandra S and Scott, Clayton},   
  journal={arXiv preprint arXiv:2011.05309},   
  year={2020}   
}

There are several helper files. The callable files are:

1) lspca_sub.m - supervised pca with the least squares loss function; for regression; uses substitution for \beta in place of alternating updates
2) lspca_MLE_sub.m - same as above; uses maximum likelihood updates for tuning parameter \lambda
3) lrpca.m - supervised pca with the logistic loss function; for classification
4) lrpca_MLE.m same as above; uses maximum likelihood updates for tuning parameter \lambda
5) the four files above prepended with 'k' - kernelized versions of those algorithms
