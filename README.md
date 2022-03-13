# weightedDataset
Paper: Deep neural learning on weighted datasets utilizing label disagreement from crowdsourcing
https://www.sciencedirect.com/science/article/pii/S1389128621002711

Experts and crowds can work together to generate high-quality datasets, but such collaboration is limited to a large-scale pool of data. In other words, training on a large-scale dataset depends more on crowdsourced datasets with aggregated labels than expert intensively checked labels. However, the limited amount of high-quality dataset can be used as an objective test dataset to build a connection between disagreement and aggregated labels. In this paper, we claim that the disagreement behind an aggregated label indicates more semantics (e.g. ambiguity or difficulty) of an instance than just spam or error assessment. We attempt to take advantage of the informativeness of disagreement to assist learning neural networks by computing a series of disagreement measurements and incorporating disagreement with distinct mechanisms. Experiments on two datasets demonstrate that the consideration of disagreement, treating training instances differently, can promisingly result in improved performance.

# file systems
## main.py
To run the experiment

## preprocessing.py
prepare the training data

## data_reader.py
supply data reading and resource loading functions, including word embeddings
