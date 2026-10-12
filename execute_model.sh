#!/bin/bash

for i in {0..2}
do
    python main.py --test Test --train Train --data toy_example_$i
done