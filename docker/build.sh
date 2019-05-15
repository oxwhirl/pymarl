#!/bin/bash

echo 'Building Dockerfile with image name pymarl:1.0'
cp ../requirements.txt .
docker build -t pymarl:1.0 .
