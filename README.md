# ASNP-RMR

This is an official PyTorch implementation of the ASNP-RMR model presented in the following paper:

> [Robustifying Sequential Neural Processes](https://proceedings.icml.cc/static/paper_files/icml/2020/4915-Paper.pdf)
> , *[Jaesik Yoon](https://sites.google.com/view/jaesikyoon/home), [Gautam Singh](https://singhgautam.github.io/), [Sungjin Ahn](https://sungjinahn.com/)*
> , *ICML 2020*

This code contains NP, ANP, SNP, ASNP-W and ASNP-RMR with ablation to compare the performance.

It is based on the [Deepmind Attentive Neural Processes code]( https://github.com/deepmind/neural-processes).

## General

backup folder is to store SNP code, and each fig folders include the codes to visualize the results as reported in the paper.

## How to use

Every model and test is starting from main.py. By configuring the hyperparameters on there, you can run. MNIST and Celeb A tests require download of the dataset, which will process automatically on the code.

## Contact

Any feedback is welcome! Please open an issue on this repository or send email to Jaesik Yoon (jaesik817@gmail.com).

