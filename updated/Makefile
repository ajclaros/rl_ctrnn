##
#
#
# @file
# @version 0.1

main: main.o CTRNN.o
	g++ -o main main.o CTRNN.o
CTRNN.o: CTRNN.cpp CTRNN.h queue.h
	g++ -c -O3 CTRNN.cpp
main.o: main.cpp CTRNN.h queue.h
	g++ -c -O3 main.cpp
clean:
	rm *.0 main


# end
