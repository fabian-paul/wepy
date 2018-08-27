CPPC = g++
CPPFLAGS = -Wall -std=c++11 -fPIC
SUFFIX = .so #`python-config --extension-suffix`

all: toymodel

toymodel.o: toymodel.cpp
	$(CPPC) -O3 $(CPPFLAGS) `python -m pybind11 --includes` -c toymodel.cpp -o toymodel.o

toymodel: toymodel.o
	$(CPPC) -lm -shared toymodel.o -o toymodel$(SUFFIX)

clean:
	rm *.o toymodel$(SUFFIX) test
