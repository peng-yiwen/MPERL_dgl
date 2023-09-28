# Baseline: RDF2Vec

This repository is an implementation of the RDF2Vec model for entity type prediction on FB15kET and YAGO43kET datasets.

- We use **[pyRDF2Vec](https://github.com/IBCNServices/pyRDF2Vec/tree/940ef534cd44698dfb625a0f55a47b781a8dacae)** for learning entity embeddings and use **[sklearn](https://scikit-learn.org/stable/)** for Linear SVM.
- Datasets are available through `data/`. For both datasets, we add the standard prefix to datasets and store the knowledge in RDF format.

## Parameter Settings:

We use the default settings

- For the embeddings, we used a learning rate of 0.025, dimension size of 500, a window size of 10, 10 SkipGram iterations and 25 negative samples. We use the **Weisfeiler-Lehman walking strategy** and uniform sampler by default to extract sequences from knowledge graphs.
- As Classifier, we used a linear SVM with l2 penalty, squared hinge loss.

## How to run:

- FB15k dataset: (best C=0.01)

```markdown
python3 [run.py](http://run.py/) --dataset FB15k --max_depth 2 --n_walks 30
```

- YAGO43k dataset: (best C=0.1)

```markdown
python3 [run.py](http://run.py/) --dataset YAGO43k --max_depth 2 --n_walks 30
```