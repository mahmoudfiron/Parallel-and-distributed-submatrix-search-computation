#include "compute.h"
#include <math.h>
#include <omp.h>

// This function calculates how well a small object matches a specific position in a larger picture. 
// It compares each pixel in the object with the corresponding pixel in the picture at position (i,j). 
// For each pixel pair, it calculates the relative difference: |picture_value - object_value| / picture_value. 
// It adds up all these differences and returns the total sum. A smaller sum means a better match. 
// If the sum is below the threshold, we consider it a successful match.
static inline double match_position(const Picture* P,const ObjectT* O,int i,int j){
 const int N=P->N, n=O->n; 
 const int* p=P->a; 
 const int* o=O->a; 
 double sum=0.0;
 for(int r=0;r<n;++r){
    int baseP=(i+r)*N+j, baseO=r*n; 
    for(int c=0;c<n;++c){
        int pv=p[baseP+c], ov=o[baseO+c]; 
        sum+=fabs((double)(pv-ov)/(double)pv);
    }
}
 return sum;
}

// This function searches through a picture to find if any of the given objects appear in it. 
// It tries each object one by one, and for each object, it checks every possible position where 
// the object could fit in the picture. It uses multiple CPU threads (OpenMP) to check many positions 
// at the same time for speed. When any thread finds a match (score below threshold), it immediately 
// stops all other threads using atomic operations to avoid race conditions. The function returns 
// true and fills in the result details if a match is found, or returns false if no objects match.
bool find_match_for_picture(const Picture* P,const ObjectT* objs,int M,double threshold,MatchResult* out){

 out->pictureId = P->id;
    out->found     = 0;
    out->objectId  = -1;
    out->posI      = -1;
    out->posJ      = -1;

    const int N = P->N;

    for (int k = 0; k < M; ++k) {
        const ObjectT* O = &objs[k];
        const int n = O->n;
        if (n > N) continue;

        const int maxI = N - n;
        const int maxJ = N - n;

        int foundFlag = 0;   // shared among tasks for this object
        int winI = -1, winJ = -1;

        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                for (int i = 0; i <= maxI; ++i) {
                    #pragma omp task firstprivate(i) shared(foundFlag, winI, winJ, P, O, threshold, maxJ, N)
                    {
                        // If someone already found a match, this task does nothing
                        if (!__atomic_load_n(&foundFlag, __ATOMIC_RELAXED)) {

                            for (int j = 0; j <= maxJ; ++j) {
                                if (__atomic_load_n(&foundFlag, __ATOMIC_RELAXED)) break;

                                double sum = match_position(P, O, i, j);
                                if (sum < threshold) {
                                    int expected = 0;
                                    if (__atomic_compare_exchange_n(&foundFlag, &expected, 1, 0,
                                                                    __ATOMIC_SEQ_CST, __ATOMIC_RELAXED)) {
                                        winI = i;
                                        winJ = j;
                                    }
                                    break; // stop scanning j once a match is seen
                                }
                            }
                        }
                    } // task
                }     // for i
            } // single
            #pragma omp taskwait
        } // parallel

        if (foundFlag) {
            out->found    = 1;
            out->objectId = O->id;
            out->posI     = winI;
            out->posJ     = winJ;
            return true; // picture done when any object matches
        }
    }

    return false; // no object matched this picture
}