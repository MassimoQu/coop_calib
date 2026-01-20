#!/bin/bash
# Define the model directory and other parameters
MODEL_DIR=$1
FUSION_METHOD=$2
# TEST_EPOCHS=('12' '14' '16' '18' '10' '54' '65' '60')  # Add more epochs as needed
# TEST_EPOCHS=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14')  # Add more epochs as needed
TEST_EPOCHS=( '4' '7' '12' '13')
# Loop through the test epochs and run the inference command
for EPOCH in "${TEST_EPOCHS[@]}"; do
    echo "Running inference for epoch $EPOCH..."
    python opencood/tools/inference.py --model_dir $MODEL_DIR --fusion_method $FUSION_METHOD --test_epoch $EPOCH
done