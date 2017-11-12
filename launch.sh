#!/bin/bash
# script making and running classificator program and then calculate precision

# make project
make

cd ./build/bin

# --------------- without SSE ---------------
# train classificator (no SSE)
echo
echo "Training (no SSE):"
echo
./task2 -d ../../data/multiclass/train_labels.txt -m model.txt --train

# test classificator (no SSE)
echo
echo Predict:
echo
./task2 -d ../../data/multiclass/test_labels.txt -m model.txt -l predict.txt --predict

cd ../..

# calculate precision
echo 
python compare.py ./data/multiclass/test_labels.txt ./build/bin/predict.txt 
echo

# --------------- using SSE ---------------
cd ./build/bin

# train classificator
echo
echo "Training (SSE):"
echo
./task2 -d ../../data/multiclass/train_labels.txt -m model.txt --train --sse

# test classificator
echo
echo Predict:
echo
./task2 -d ../../data/multiclass/test_labels.txt -m model.txt -l predict.txt --predict --sse

cd ../..

#compare result
echo
python compare.py ./data/multiclass/test_labels.txt ./build/bin/predict.txt 
echo