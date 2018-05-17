The `src` folder contains the implementations of **Skip-Gram**, **Bayesian Skip-Gram** and **Embed Align**.

### To train
1. Set the parameters in the respective parameters file {SkipGram, BayesianSkipGram, EmbedAlign}_parameters.py.
2. Choose the datafile. The default implementation runs the `dev.en` files.
3. `python *_main.py` trains the respective model and saves the file in `../models` folder.

### To evaluate
1. `python *_evaluate.py` Creates the `.out` files in `../results` folder.
