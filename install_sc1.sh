#!/bin/bash
# Install SC1 from the oxwhirl/starcraft_ubuntu repo 

git clone git@github.com:oxwhirl/starcraft_ubuntu.git sc1_temp
mkdir -p 3rdparty
mv sc1_temp/StarCraftI 3rdparty
rm -rf sc1_temp
