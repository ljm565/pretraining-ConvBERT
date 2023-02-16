#!/bin/sh

## setup
dpath=../../data/kowikitext/processed
tpath=../../data/kowikitext/tokenizer
vocab_size=30000

## train the vocab
mkdir $tpath
mkdir $tpath/vocab_$vocab_size
python3 vocab_trainer.py --data $dpath/kowikitext.all --size $vocab_size --output $tpath/vocab_$vocab_size