##
#
#
# @file
# @version 0.1

main: main.o CTRNN.o microbial.o LeggedAgent.o
	g++ -o main main.o CTRNN.o microbial.o LeggedAgent.o
microbial.o: microbial.cpp microbial.h CTRNN.h
	g++ -c -O3 microbial.cpp
CTRNN.o: CTRNN.cpp CTRNN.h queue.h
	g++ -c -O3 CTRNN.cpp
LeggedAgent.o: LeggedAgent.cpp LeggedAgent.h CTRNN.h
	g++ -c -O3 LeggedAgent.cpp

main.o: main.cpp CTRNN.h microbial.h LeggedAgent.h
	g++ -c -O3 main.cpp
clean:
	rm *.0 main


# end
