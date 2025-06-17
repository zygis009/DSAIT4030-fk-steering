#!/bin/bash

python launch_eval_runs.py --use_smc --model_name='stabilityai/stable-diffusion-xl-base-1.0' --lmbda=10.0 --resample_frequency=20 --resample_t_start=20 --resample_t_end=80 --num_particles=4 --potential_type=max