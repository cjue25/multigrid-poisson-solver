#usage: sh compile.sh filename.cpp

# filename
IN=${1}

# your compiler
CC=g++-7

# openmp flag of your compiler
OPENFLAG=-fopenmp
#OPENFLAG=-openmp
#OPENFLAG=-qopenmp

# set the output name
OUT=$(basename ${IN} .cpp).out

${CC} ${OPENFLAG} ${IN} -o ${OUT}
