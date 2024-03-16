# Codebase for BSc Thesis "GAN Based Generation of Multimodal Time Series Synthetic Data from Remotely Monitored Diabetes Patients"

## Quickstart Guide

This guide will walk you through the process of extracting data from [Awesome-CGM](https://github.com/irinagain/Awesome-CGM/wiki/Colas-(2019)) repository, setting up your environment, compiling the data, and training a GAN model.

### Extract Data

First, you need to extract the data from the [Awesome-CGM](https://github.com/irinagain/Awesome-CGM/wiki/Colas-(2019)) repository to the `data/online` directory.

### Set Up Environment

Ensure you have the following set up:

- Virtual Environment (venv)
- PyTorch
- CUDA

For detailed instructions, refer to `useful_scripts.txt`.

### Compile Data

Run `compiler-online.py` to compile the extracted data. Refer to the commented code in `compiler-dialect.py` for guidance on this step.

### Encode Data

Once compiled, use `processing-pipeline-healthgan.py` to encode the compiled data. Again, you can refer to the commented code in `compiler-dialect.py` for assistance.

### Train GAN

Place the resulting sdv file from the encoding step into `data/train` and proceed to train your GAN model.

By following these steps, you should be able to set up your environment, compile and encode the data, and train your first GAN model. Don't forget to decode the synthetic data after training.

Have fun! 🚀
\- Mike
