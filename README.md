# Rainbow Plots
These scripts will align course description text to ACM/IEEE curriculum topic level maps, prepare anonymous student enrollment data of courses over time, and then generate rainbow barcode plots to show how student topic coverage accumulates over time. The methods used are CS specific, where a word-to-vec style embedding model trained by [Efstathiou et al](https://github.com/vefstathiou/SO_word2vec) was leveraged to establish second-order features used to match ACM topics to course description text. This model is simple and parsimonious, and works fairly well for obvious cases, but struggles with aligning topics beyond the vocabulary of the embedding. This project serves more as a template for processing and preparing these style of curriculum visualizations than for the NLP techniques.

## Installation

