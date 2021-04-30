########################## Modified KMAC V1.0 ##########################

(The efficient O(n log n) implementation of 2D and 3D Expected
Hypervolume Improvement (EHVI))

Original C++ code by Michael Emmerich, Leiden University

Modifications and pybind11 wrapper coded by Tim Jim, Tohoku University
21/04/2021

########################################################################

TL;DR --> install like this:
   ($ activate kadal_env / $ conda activate kadal_env)  # if necessary
   $ pip install pybind11 / $ conda install -c conda-forge pybind11
   $ cmake .
   $ make
   $ python test_single.py


########################################################################

This directory contains:

 - A modified version of the KMAC C++ source code
   -- Obtained from: https://moda.liacs.nl/index.php?page=code
   -- The code is from the "EHVI/EHVI_3D/HDYE_3D_Update" directory.
   -- N.B. The 2D version has not been wrapped yet.
   -- The original implementation did not clean up pointers after use,
      so leaked memory.
   -- All raw pointers have been replaced with shared pointers as per:
      https://stackoverflow.com/questions/67110549/pybind11-identify-and-remove-memory-leak-in-c-wrapper
   -- The code is then recompiled in C++ and checked with valgrind to
      ensure any memory leaks have been removed.
   -- Only the code paths following the 'sliceupdate' method have been
      properly checked.
   -- A set of test input files has also been included
      (test_single.txt, test_multi.txt)
   -- If you choose to compile the C++ executable for testing/curiosity,
      you can check with these input files (optional - not required
      for the python wrapper).
   -- e.g. (set mtune depending on your processor)
   
      $ g++ -Ofast -o EHVI -std=c++0x -mtune=corei7 *.cc
      $ ./EHVI test.txt sliceupdate
      
   -- Or use the original executables EHVI (renamed EHVI_original),
      EHVI.exe, hdye_3d.mexw64, which have been left in also for your
      testing purposes.


 - The pybind11 C++ wrapper (kmac_wrapper.cpp) and CMakeLists.txt
   -- The warpper defines the functions to be exposed to Python
   -- CMakeLists.txt specifies the location of files required for
      the creation of a importable shared object file.


 - python test files (test_single.py, test_multi.py)
   -- Once the kmac shared object library is built, run the test files
   -- Check they can import the kmac library, the assert passes, and
      RAM usage is stable.
   -- If python complains with an ImportError that kmac is missing,
      double check you have run the cmake compilation with the python
      version you intend to use. (i.e. check `which python` matches
      your expectations.)


########################## BULDING THE WRAPPER LIBRARY ##########################

 - Double-check the paths/version requirements in CMakeLists.txt look
   reasonable for your environment.
   
 - Activate your python evironment (double-check with `which python`)
   -- Check you have installed pybind11 in this env
   -- e.g.

      $ pip install pybind11
      or:
      $ conda install -c conda-forge pybind11
   -- Or, just clone the pybind11 repo from github into this directory.
      (In this case, comment/uncomment the relevant lines in
      CMakeLists.txt, under the "# Load pybind11" heading.)

 - In this directory, run:
 
      $ cmake .
      $ make

   -- This should output a new kmac.*.so file corresponding to your
      python version.
      
 - Run the python test files:
   -- e.g.
   
      $ python test_single.py
      $ python test_multi.py

   -- The answers should correspond with running the original EHVI
      application with test_single.txt and test_multi.txt.
   -- N.B. A infinite loop is run in the test files, so you can open
      up your system resource monitor and check the memory is not
      increasing indefinitely. This is important, as the EHVI wrapper
      is run in a tight loop by KADAL for each EHVI function evaluation
      (on each population member/sample).

 - If you have issues or compiled with the wrong python version/get
   strange errors, it can sometimes help to remove the CMakeCache.txt
   file (and maybe CMakeFiles dir) before trying `cmake .` and `make`
   again.
