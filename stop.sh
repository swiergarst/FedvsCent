#!/bin/bash


if [ $1 = "IID" ]
then
  NODE="IID_org"
fi

if [ $1 = "s" ]
then
  NODE="sample_imb_org"
fi
if [ $1 = "c" ]
then
  NODE="class_imb_org"
fi
if [ $1 = "f" ]
then
  NODE="fashion_MNIST_org"
fi
if [ $1  = "a" ] 
then
  NODE="A2_org"
fi

if [ $1 -eq 3 ]
then
  NODE="3node_org"
  CLIENTS=3
elif [ $1 -eq 2 ]
then
  NODE="2node_org"
  CLIENTS=2
else
  CLIENTS=10
fi


for ((i = 0; i<=$CLIENTS-1; i++))
do 
      vnode stop --name ${NODE}${i} 
done

docker volume prune -f
docker network prune -f
