#!/bin/bash

# Developed by : Sameera Abeykoon (January 2018)
# This script will take all the nii.gz files and
# put each file in seperate directory and unzip the
# all the files 


# Please give the directory path
cd $1

for x in *.nii.gz; do
  echo "$x"
  
  # Find the first 6 charcters of the data file
  d=$(echo "$x" | cut -c1-6)
  echo "$d"
  
  # make a seperate dir for each data file and move the 
  # each .nii.gz file to that directory
  mkdir -p "$d"
  mv -- "$x" "$d/"
  
  # change to new directory
  cd $PWD"/$d"
  echo $PWD

  # extract all the files using fsl command fslsplit
  fslsplit "$x" "$d" -t

  # go back to the main directory where all other .nii.gz contain
  cd ../
  echo $PWD
done

