#!/bin/bash


prefix=${PWD}/vantage6_config_files/node/
case $1 in
 
  "IID")
    if [ $2 -eq 2 ]
    then
      NODE="2class_IID/IID_org"
    elif [ $2 -eq 4 ]
    then
      NODE="4class_IID/IID_org"
    fi
    ;;
    "s")
      if [ $2 -eq 2 ]
      then
        NODE="2class_sample_imb/sample_imb_org"
      elif [ $2 -eq 4 ]
      then
        NODE="4class_sample_imb/sample_imb_org"
      fi
    ;;
    "c")
      if [ $2 -eq 2 ]
      then
        NODE="2class_class_imb/class_imb_org"
      elif [ $2 -eq 4 ]
      then
        NODE="4class_class_imb/class_imb_org"
      fi
    ;;
  "f")
    if [ $2 == c ]
    then
      NODE="fashion_MNIST_ci/fashion_MNIST_org"
    else
      NODE="fashion_MNIST/fashion_MNIST_org"
    fi
      ;;
  "a")
    case $2 in
      "IID")
        if [ $3 == p ]
        then
          NODE="A2_IID/PCA/A2_org"
        elif [ $3 == r ]
        then
          NODE="A2_IID/raw/A2_org"
        fi
      ;;
    "s")
        if [ $3 == p ]
        then
          NODE="A2_sample_imb/PCA/A2_org"
        elif [ $3 == r ]
        then
          NODE="A2_sample_imb/Raw/A2_org"
        fi
        ;; 
    "c")
        if [ $3 == p ]
        then
          NODE="A2_class_imb/PCA/A2_org"
        elif [ $3 == r ]
        then
          NODE="A2_class_imb/Raw/A2_org"
        fi
    esac
esac

if [ $1 -eq 3 ]
then
  if [ $2 == r ]
  then
    NODE="3node_raw/3node_org"
  elif [ $2 == p ]
  then
    NODE="3node_PCA/3node_org"
  elif [ $2 == b ]
  then
    NODE="3node_PCA_balanced/3node_org"
  fi
  CLIENTS=3
  elif [ $1 -eq 2 ]
  then
    NODE="2node_PCA/2node_org"
    CLIENTS=2
  else
    CLIENTS=10
fi
  
for ((i = 0; i<=$CLIENTS-1; i++))
do 
      vnode start --config "$prefix$NODE$i.yaml" -e dev --image harbor2.vantage6.ai/infrastructure/node:2.1.0.post1
done


