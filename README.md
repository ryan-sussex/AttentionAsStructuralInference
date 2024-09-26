# Attention as Structural Inference

Code related to the [paper: Attention as Structural Inference](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4e8a74988bc611495c2d3a5edac8493f-Abstract-Conference.html)

## Expanding Attention
Expanding attention adjusts the implicit uniform prior two be geometrically decaying, with a conjugate beta hyperprior.

`python ./run_expanding.py`

## Multihop attention
Searches through paths of length two in the adjacency graph to decide what to attend to.

`python ./run_multihop.py`

Other files
---
`models.py` contains neural net definitions

`data.py` and `data_expanding.py` contain the code for generating toy data as described in the paper.
