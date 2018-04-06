#!/bin/bash

# kubectl create -f clear-data.yaml

# 'select_max', 'abs', 'relu', 'select'
kubectl create -f select-max-lr.yaml
kubectl create -f abs-lr.yaml
kubectl create -f relu-lr.yaml
kubectl create -f select-lr.yaml
