#usage: sh compile.sh filename.cpp

# filename
IN=${1}

# your compiler
CC=mpic++

# set the output name
OUT=$(basename ${IN} .cpp).out

${CC} ${OPENFLAG} ${IN} -o ${OUT}

