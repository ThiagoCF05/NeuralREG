# NeuralREG

This project provides the data and models described on the ACL 2018 paper "NeuralREG: An end-to-end approach to referring expression generation"
(available [here](https://www.aclweb.org/anthology/papers/P/P18/P18-1182/)).

## NeuralREG models

[**NeuralREG+Seq2Seq**](seq2seq.py)

Seq2Seq version. To train and evaluate the model, you may update the variable paths in the script and run 
the following command:

```python
python seq2seq.py --dynet-gpu
```

[**NeuralREG+CAtt**](attention.py) 

Concatenative attention version. To train and evaluate the model, you may update the variable paths 
in the script and run the following command:

```python
python3 attention.py --dynet-gpu
```

[**NeuralREG+HierAtt**](hierattention.py)

Hierarcical attention version. To train and evaluate the model, you may update the variable paths in the script and 
run the following command:

```python
python hierattention.py --dynet-gpu
```

## Data

[**WebNLG**](webnlg/)

The original and delexicalized versions of the WebNLG corpus used in our experiments.

[**Referring Expressions**](data/)

Training, development and test referring expressions sets and vocabularies. This is the official data used to train and 
evaluate the models. It was extracted from [WebNLG/](webnlg/) using the command:

```bash
python preprocessing.py [IN_PATH] [OUT_PATH] [STANFORD_PATH]
```

## Baselines

[**OnlyNames**](only_names.py)

*OnlyNames* baseline. The model may be executed by the following command:

```python
python2.7 only_names.py
```

[**Castro Ferreira et al.**](ferreira)

This baseline is an adaptation of the model described in [this paper](http://www.aclweb.org/anthology/P16-1054).
The model may be executed by the following commands:

```python
python2.7 reg_train.py
python2.7 reg_main.py
```

## Evaluation



[**eval/**](eval) 

Automatic evaluation scripts to extract information about the referring expression collection (corpus.py), to obtain the results depicted in the paper (evaluation.py) and to test statistical significance (statistics.R)

[**humaneval/**](humaneval) 

Human evaluation scripts to obtain results depicted in the paper (stats.py) and to test statistical significance (statistics.R)

## Citation

```Tex
@InProceedings{ferreiraetal2018b,
  author = 	"Castro Ferreira, Thiago
		and Moussallem, Diego
		and K{\'a}d{\'a}r, {\'A}kos
		and Wubben, Sander
		and Krahmer, Emiel",
  title = 	"NeuralREG: An end-to-end approach to referring expression generation",
  booktitle = 	"Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = 	"2018",
  publisher = 	"Association for Computational Linguistics",
  pages = 	"1959--1969",
  location = 	"Melbourne, Australia",
  url = 	"http://aclweb.org/anthology/P18-1182"
}
```

**Author:** Thiago Castro Ferreira

**Date:** 15/12/2017 (*Updated on June 3rd 2019*)
