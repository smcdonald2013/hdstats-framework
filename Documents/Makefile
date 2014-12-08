CFLAGS= -Wall

all : duffing_euler duffing_ab2 duffing_rk4

duffing_euler: euler.o duffing.o
	cc -o duffing_euler euler.o duffing.o -lm $(CFLAGS)
euler.o duffing.o : integrator.h

duffing_ab2 : ab2.o duffing.o
	cc -o duffing_ab2 ab2.o duffing.o -lm $(CFLAGS)
ab2.o duffing.o : integrator.h

duffing_rk4 : rk4.o duffing.o
	cc -o duffing_rk4 rk4.o duffing.o -lm $(CFLAGS)
rk4.o duffing.o : integrator.h  



