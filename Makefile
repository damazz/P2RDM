EIGENDIR = /home/andrew/eigen
LIBINTDIR = /home/andrew/libint
LIBINT_LINK = -std=c++11 -isystem ${LIBINTDIR}/include -isystem ${LIBINTDIR}/include/libint2 -L${LIBINTDIR}/lib -DSRCDATADIR=\"${LIBINTDIR}/share/libint/2.1.0-stable/basis\"

# include headers the object include directory
#CPPFLAGS += -std=c++11 -O3 -I./  -isystem ${EIGENDIR}/ ${LIBINT_LINK} 
CPPFLAGS += -DARMA_64BIT_WORD -std=c++11 -O3 -I./  -isystem ${EIGENDIR}/ ${LIBINT_LINK} 
#CPPFLAGS += -Wall -Wno-sign-compare -Wunused-but-set-variable -std=c++11 -march=native -ffast-math -fopenmp -msse2 -O3 -I./  -isystem ${EIGENDIR}/ ${LIBINT_LINK} 
#CPPFLAGS += -Wall -Wno-sign-compare -Wunused-but-set-variable -std=c++11 -march=native -ffast-math -O2 -I./  -isystem ${EIGENDIR}/ ${LIBINT_LINK}
CXXPOSTFLAGS = -lint2 -larmadillo

main.exe : main.o sym.o D2.o HF.o ap.o alglibinternal.o alglibmisc.o linalg.o solvers.o optimization.o 
	g++ ${CPPFLAGS} sym.o D2.o HF.o ap.o alglibinternal.o alglibmisc.o linalg.o solvers.o optimization.o main.o -o main.exe -larmadillo ${CXXPOSTFLAGS}
D2.o : D2.cpp D2.h
	g++ -c D2.cpp
HF.o : HF.cpp HF.h D2.h
	g++ ${CPPFLAGS} -c HF.cpp
sym.o : sym.cpp sym.h
	g++ ${CPPFLAGS} -c sym.cpp 
ap.o : ap.cpp ap.h stdafx.h
	g++ -c ap.cpp
alglibinternal.o : alglibinternal.cpp alglibinternal.h ap.h
	g++ -c alglibinternal.cpp
alglibmisc.o : alglibmisc.cpp alglibmisc.h
	g++ -c alglibmisc.cpp
linalg.o : linalg.cpp linalg.h
	g++ -c linalg.cpp
solvers.o : solvers.cpp solvers.h
	g++ -c solvers.cpp
optimization.o : optimization.cpp optimization.h
	g++ -c optimization.cpp
main.o : main.cpp optimization.h HF.h 
	g++ ${CPPFLAGS} -c main.cpp ${CXXPOSTFLAGS}

clean:
	-rm *.o 
