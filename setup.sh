!#/bin/bash

wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh
chmod u+x Anaconda2-4.3.1-Linux-x86_64.sh
./Anaconda2-4.3.1-Linux-x86_64.sh -b

# Create the ML_Tutorial environment
conda create –f ML_Tutorial.yml

# Activate the environment
source activate ML_Tutorial

# Close the environment once done
# source deactivate
