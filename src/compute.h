#pragma once
#include <stdbool.h>
#include "types.h"
bool find_match_for_picture(const Picture* pic,const ObjectT* objs,int M,double threshold,MatchResult* out);
