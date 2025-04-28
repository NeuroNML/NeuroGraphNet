#!/usr/bin/env bash

# Tunnel remote jupyter kernel to localhost
# ssh -L <PORT NUMBER>:<IP ADDRESS>:<PORT NUMBER> -l <USERNAME> <front-node> -f -N
ssh -L 8306:10.91.27.54:8306 -l ldibello izar.hpc.epfl.ch -f -N
