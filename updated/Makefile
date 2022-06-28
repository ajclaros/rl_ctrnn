##
#
#
# @file
# @version 0.1

main: main.o CTRNN.o microbial.o LeggedAgent.o RLCTRNN.o auxilary.o
	g++ -o main main.o CTRNN.o microbial.o LeggedAgent.o RLCTRNN.o auxilary.o
microbial.o: microbial.cpp microbial.h CTRNN.h auxilary.h
	g++ -c -O3 microbial.cpp
CTRNN.o: CTRNN.cpp CTRNN.h auxilary.h
	g++ -c -O3 CTRNN.cpp
RLCTRNN.o: CTRNN.h RLCTRNN.cpp RLCTRNN.h queue.h auxilary.h
	g++ -c -O3 RLCTRNN.cpp
auxilary.o: auxilary.cpp auxilary.h
	g++ -c -O3 auxilary.cpp
LeggedAgent.o: LeggedAgent.cpp LeggedAgent.h CTRNN.h auxilary.h
	g++ -c -O3 LeggedAgent.cpp

main.o: main.cpp CTRNN.h microbial.h LeggedAgent.h auxilary.h RLCTRNN.h queue.h
	g++ -c -O3 main.cpp
clean:
	rm *.o main
# end
