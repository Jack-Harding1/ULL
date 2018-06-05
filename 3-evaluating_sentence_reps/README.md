
### Introduction

In this project, we experiment with and evaluate sentence representations. Specifically, we test the performance of (learned) Skip-gram (vanilla [1] and Bayesian [2]) and Embed-Align [3] models on sentence representations. We use SentEval [4], an evaluation toolkit by facebook for sentence embeddings.

### Sentence embeddings

We experiment with the following sentence embeddings:
1. Mean of word embeddings
2. Sum of word embeddings
3. Element-wise multiplication of word embeddings
4. Weighted average (weights based on inverse frequency of the words in the corpora)
5. Random average

### Requirements

For training the skip-gram model, we use the `Word2Vec` function provided in gensim. We use Europarl [5] data for training.

### Running the code

We prepreocess the data to remove stopwords, and to restrict the vocabulary size. The parameters can be set in `preprocess.py`. The new dataset is created in the path `/data/europarl/`.

Each of the files in src folder can be run independently `python <file_name>.py`. The code should run without errors if the data files are provided in the file structure mentioned [here](data).

### Models and Results

Trained models should be put in `/models/*`. Results are created in `/results/` folder.

### References

1. Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. **Efficient estimation of word representations in vector space**. arXiv preprint arXiv:1301.3781, 2013.

2. Arthur Braˇzinskas, Serhii Havrylov, and Ivan Titov. **Embedding words as distributions with a bayesian skip-gram model**. arXiv preprint arXiv:1711.11027, 2017.

3. Miguel Rios, Wilker Aziz, and Khalil Sima’an. **Deep generative model for joint alignment and word representation**. arXiv preprint arXiv:1802.05883, 2018.

4. Alexis Conneau and Douwe Kiela. 2018. **Senteval: An evaluation toolkit for universal sentence representations**. arXiv:1803.05449v1.

5. Philipp Koehn. 2005. **Europarl: A parallel corpus for statistical machine translation**. MT Summit.
