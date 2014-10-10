#!/bin/bash
for i in `seq $1 10000 100000`; do
  time ./test.sh ibm2p test $i french -Xmx4g > "log/${i}.${2}.log"
done
