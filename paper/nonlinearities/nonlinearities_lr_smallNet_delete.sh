#!/bin/bash

# kubectl create -f clear-data.yaml

# 'select_max', 'abs', 'relu', 'select'
kubectl delete jobs/select-max-lr-01
kubectl delete jobs/select-max-lr-001
kubectl delete jobs/select-max-lr-0001
kubectl delete jobs/select-max-lr-00001

kubectl delete jobs/abs-lr-01
kubectl delete jobs/abs-lr-001
kubectl delete jobs/abs-lr-0001
kubectl delete jobs/abs-lr-00001

kubectl delete jobs/relu-lr-01
kubectl delete jobs/relu-lr-001
kubectl delete jobs/relu-lr-0001
kubectl delete jobs/relu-lr-00001

kubectl delete jobs/select-lr-01
kubectl delete jobs/select-lr-001
kubectl delete jobs/select-lr-0001
kubectl delete jobs/select-lr-00001
