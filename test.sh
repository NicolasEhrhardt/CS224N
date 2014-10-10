#!/bin/bash
data="data/"

case "$1" in
  base)
    java -cp pa1/java/classes cs224n.assignments.WordAlignmentTester \
    -dataPath $data \
    -model cs224n.wordaligner.BaselineWordAligner \
    -evalSet $2 \
    -trainSentences ${3:-200} \
    -language ${4:-"french"} \
    ${@:5}
    ;;
  pmi)
    java -cp pa1/java/classes cs224n.assignments.WordAlignmentTester \
    -dataPath $data \
    -model cs224n.wordaligner.PMIModel \
    -evalSet $2 \
    -trainSentences ${3:-200} \
    -language ${4:-"french"} \
    ${@:5}
    ;;
  ibm1)
    java -cp pa1/java/classes cs224n.assignments.WordAlignmentTester \
    -dataPath $data \
    -model cs224n.wordaligner.IBMModel1 \
    -evalSet $2 \
    -trainSentences ${3:-200} \
    -language ${4:-"french"} \
    ${@:5}
    ;;
  ibm2)
    java -cp pa1/java/classes cs224n.assignments.WordAlignmentTester \
    -dataPath $data \
    -model cs224n.wordaligner.IBMModel2 \
    -evalSet $2 \
    -trainSentences ${3:-200} \
    -language ${4:-"french"} \
    ${@:5}
    ;;
  ibm2p)
    java -cp pa1/java/classes cs224n.assignments.WordAlignmentTester \
    -dataPath $data \
    -model cs224n.wordaligner.IBMModel2Parallel \
    -evalSet $2 \
    -trainSentences ${3:-200} \
    -language ${4:-"french"} \
    ${@:5}
    ;;
     
  *)
    echo "Usage: $0 {mini,pmi} {dev,miniTest} {number}"
    exit 1
esac

