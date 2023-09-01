#include <iostream>
#include <iomanip>

#define WAPRSIZE 32
#define MTXOFF(row, ld, col) ((row) * (ld) + (col))

#ifndef executeType
enum executeType {
    EXECcuBLAS=1, 
    EXECtileGemm=2
};
#endif

