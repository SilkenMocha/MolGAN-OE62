#!/bin/bash
wget -O OE62_dataset.zip https://dataserv.ub.tum.de/index.php/s/m1507656/download
unzip OE62_dataset.zip
rm OE62_dataset.zip

wget https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
wget https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz


