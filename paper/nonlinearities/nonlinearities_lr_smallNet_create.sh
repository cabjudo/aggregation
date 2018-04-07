#!/bin/bash

# kubectl create -f clear-data.yaml

# 'select_max', 'abs', 'relu', 'select'
kubectl create -f /home/christine/projects/convnet/paper/nonlinearities/select-max-lr.yaml
kubectl create -f /home/christine/projects/convnet/paper/nonlinearities/abs-lr.yaml
kubectl create -f /home/christine/projects/convnet/paper/nonlinearities/relu-lr.yaml
kubectl create -f /home/christine/projects/convnet/paper/nonlinearities/select-lr.yaml
