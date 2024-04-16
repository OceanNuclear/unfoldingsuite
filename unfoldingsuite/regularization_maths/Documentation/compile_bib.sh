#!/bin/sh
bibfile=$(ls *.bib |head -n1)
for i in $(ls ${1}/*.bib)
do
    cat ${i} >> $bibfile
    echo compiled ${i}
done