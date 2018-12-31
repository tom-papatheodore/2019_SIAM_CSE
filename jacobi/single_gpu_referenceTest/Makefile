CCOMP = pgcc
CFLAGS = -acc -Minfo=acc -ta=tesla:cc70 -fast

run: poisson2d.o poisson2d_serial.o
	$(CCOMP) $(CFLAGS) poisson2d.o poisson2d_serial.o -o run

poisson2d.o: poisson2d.c
	$(CCOMP) $(CFLAGS) -c poisson2d.c

poisson2d_serial.o: poisson2d_serial.c
	$(CCOMP) $(CFLAGS) -c poisson2d_serial.c

.PHONY: clean cleanall

clean:
	rm -f run *.o

cleanall:
	rm -f *.nvvp *.prof *.err *.out

