#pragma once
#include <stdbool.h>
#include "types.h"
bool read_input(const char* path,double* t,Picture** pics,int* P,ObjectT** objs,int* M);
bool write_output(const char* path,const MatchResult* r,int P);
