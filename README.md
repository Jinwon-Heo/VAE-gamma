# VAE-gamma
Gamma-mixture variational autoencoder for clustering right-skewed gene expression data
# Introduction
Our proposed model, VAE-gamma, is a clustering model for right-skewed data such as gene expression data. Because the model applies the variational autoencoder, we can reduce the dimension of data into the latent variable $z$. Thus, in high-dimensional having small sample size data, the proposed model can show good performance by using $z$ that has the low-dimensional important feature. In addition, the proposed method can model right-skewed data by adopting the gamma distribution to the generated data distribution and marginal distribution of $z$. the proposed method is applied to synthetic data, two high-dimensional real gene expression datasets, and single-cell RNA-seq data having small sample sizes. It shows better performance than the existing generative models, including statistical model-based methods such as mixtures of skewed t-factor analyzers. These files are Version 1 code of the proposed model.
# Example
Here, we generate 100 data for each group, the parameters are the same as those of the paper. $\delta$ is fixed to 0.1. If you want to use the .py, you can run VAE_gamma_simul.py after downloading all code files; if you wan to use the .ipynb, you can follow simul_example.ipynb.
# Model structure
![model structure2](https://user-images.githubusercontent.com/115125560/196144068-f57dc30e-7c58-4739-92a2-094656ae5d37.png)
