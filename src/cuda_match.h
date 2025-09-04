#pragma once
#include <stdbool.h>
#include "types.h"
#ifdef __cplusplus
extern "C" {
#endif

// Try to find a match using the GPU.
// Returns 1 if found (fills 'out'), else 0 so caller can fall back to CPU.
int cuda_find_match_for_picture(const Picture* P,
                                const ObjectT* objects, int M,
                                double threshold,
                                MatchResult* out);

#ifdef __cplusplus
}
#endif
