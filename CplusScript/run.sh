#!/bin/bash
export DYLD_LIBRARY_PATH=/opt/intel/mkl/lib:/opt/intel/lib
./a.out
rm a.out
