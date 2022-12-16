#!/bin/bash

if [ -z $1 ]
then
  echo "Missing scale factor as argument i.e. \"bash generate_data.sh 1\""
else
  cd data/
  ./dbgen -T L -s $1 -b dists.dss
fi
