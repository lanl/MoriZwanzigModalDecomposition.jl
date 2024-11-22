#!/bin/bash

for ((i=10; i <= 220; i+=10)); do
    julia ./gen_x0_errors_sweep_k_dominant_modes_pred.jl $i
done
