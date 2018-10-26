#!/bin/bash
# Install SC2 and add the custom maps

if [ -z "$EXP_DIR" ]
then
    EXP_DIR=~
fi

echo "EXP_DIR: $EXP_DIR"
cd $EXP_DIR/pymarl

mkdir 3rdparty
cd 3rdparty

export SC2PATH=`pwd`'/StarCraftII'
echo 'SC2PATH is set to '$SC2PATH

if [ ! -d $SC2PATH ]; then
        echo 'StarCraftII is not installed. Installing now ...';
        wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.1.2.60604_2018_05_16.zip
        unzip -P iagreetotheeula SC2.4.1.2.60604_2018_05_16.zip
        rm -rf SC2.4.1.2.60604_2018_05_16.zip
else
        echo 'StarCraftII is already installed.'
fi

echo 'Adding custom maps.'
MAP_DIR="$SC2PATH/Maps/Melee/"
echo 'MAP_DIR is set to '$MAP_DIR

if [ ! -d $MAP_DIR ]; then
        mkdir -p $MAP_DIR
fi

cd ..
cp src/envs/starcraft2/maps/* $MAP_DIR

echo 'StarCraftII is installed.'


