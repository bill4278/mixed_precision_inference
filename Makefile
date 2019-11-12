SRC=mixed_precision_inference
INC=-I/usr/local/cuda/include
FLAGS=-g -O0 -std=c++11
LIB=-lnvinfer -L/usr/local/cuda/lib64 -lcudart

all:
	g++ $(FLAGS) $(INC) ${SRC}.cpp $(LIB) -o $(SRC)

.PHONY:clean
clean:
	rm $(SRC)
