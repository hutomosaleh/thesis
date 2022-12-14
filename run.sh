#!/bin/bash

PS3='Choose target: '
options=("Q6" "Unified Queue" "Quit")
select opt in "${options[@]}"
do
  case $opt in
    "Q6")
      TARGET="cpp_tpch_q6"
      break
      ;;
    "Unified Queue")
      TARGET="unified_queue"
      break
      ;;
    "Quit")
      exit 1
      ;;
    *) echo "invalid option $REPLY";;
  esac
done

PS3='Choose command: '
options=("Build" "Run" "Profile" "Clean")
select opt in "${options[@]}"
do
  case $opt in
    "Build")
      cd $TARGET/build/ && cmake .. && make && cd - 
      break
      ;;
    "Run")
      ./$TARGET/build/src/main $1 
      break
      ;;
    "Profile")
      sudo nvprof ./$TARGET/build/src/main $1 
      break
      ;;
    "Clean")
      cd $TARGET/build/ && make clean && cd - 
      break
      ;;
    *) echo "invalid option $REPLY";;
  esac
done
