Overview
========

I implemented two versions of each model: one version not optimized, and a second optimized version (with the "Parallel" suffix). The optimized version is multi-threaded and its efficiency will depend on your machine. Modify the variable 'njobs' according to the number of cores you have.

Run
===

I created two script for conveniency:

    - test.sh: script to test the algorithm with in the given parameters
    - testall.sh: generates the learning curves for the extra question.

To run: make sure that the variable $data in the script 'test.sh' is set to the right location.

Note
====

'log/' contains the log files of my last run to create the learning curves.
