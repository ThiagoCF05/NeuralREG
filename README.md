# NeuralREG

This project provides the data and models described on the paper "NeuralREG: An end-to-end approach to referring expression generation".

## FILES and DIRECTORIES:

**webnlg/** 

Original and Delexicalized versions of WebNLG corpus

**preprocessing.py** 

Script for extracting the referring expression collection from the WebNLG corpus. Update the variable paths in the script and run the command:

`python2.7 preprocessing.py`

**data/** 

Training, development and test referring expressions sets and vocabularies.

**only_names.py** OnlyNames model

Update variable paths in the script and run the following command:

`python2.7 only_names.py`

**ferreira/** [Ferreira model](http://www.aclweb.org/anthology/P16-1054)

Update variable paths in the scripts and execute them in the following order to train the model and to generate the referring expressions:

```
python2.7 reg_train.py

python2.7 reg_main.py
```

**seq2seq.py** NeuralREG+Seq2Seq model 

Update the variable paths in the script and run the following command:

`python3 seq2seq.py --dynet-autobatch 1 --dynet-mem 8192 --dynet-gpu`

**attention.py** NeuralREG+CAtt model 

Update the variable paths in the script and run the following command:

`python3 attention.py --dynet-autobatch 1 --dynet-mem 8192 --dynet-gpu`

**hierattention.py** NeuralREG+HierAtt model

Update the variable paths in the script and run the following command:

`python3 hierattention.py --dynet-autobatch 1 --dynet-mem 8192 --dynet-gpu`

**eval/** 

Automatic evaluation scripts to extract information about the referring expression collection (corpus.py), to obtain the results depicted in the paper (evaluation.py) and to test statistical significance (statistics.R)

**humaneval/** Human evaluation scripts to obtain results depicted in the paper (stats.py) and to test statistical significance (statistics.R)

**Citation:**

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

**Date:** 15/12/2017
