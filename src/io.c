#include "io.h"
#include <stdio.h>
#include <stdlib.h>

// This helper function reads N*N integer numbers from a file and stores them in an array. 
// It reads the numbers one by one in row-major order (left to right, top to bottom) just like 
// reading text. If any number fails to read properly, it returns 0 for failure, otherwise 
// returns 1 for success. This is used to read both picture and object matrices from the input file.
static int read_matrix(FILE* f,int N,int* out){
    for(int i=0;i<N*N;++i){
        if(fscanf(f,"%d",&out[i])!=1)
        return 0;
    }
    return 1;
}

// This function reads the entire input file and creates all the data structures needed for the program. 
// It first reads the threshold value, then the number of pictures and all picture data (ID, size, and 
// matrix values). Next it reads the number of objects and all object data. For each picture and object, 
// it allocates memory for the matrix and calls read_matrix to fill in the values. If anything goes 
// wrong during reading, it cleans up memory and returns false. On success, it returns pointers to 
// all the loaded data and returns true.
bool read_input(const char* path,double* t,Picture** pics,int* P,ObjectT** objs,int* M){
 FILE* f=fopen(path,"r"); 
 if(!f){
    fprintf(stderr,"Failed to open input file: %s\n",path);
    return false;
}
 if(fscanf(f,"%lf",t)!=1){
    fprintf(stderr,"Failed to read threshold\n");
    fclose(f);
    return false;
}
 int p; 
 if(fscanf(f,"%d",&p)!=1){
    fprintf(stderr,"Failed to read number of pictures\n");
    fclose(f);
    return false;
}
 Picture* arr=(Picture*)calloc(p,sizeof(Picture)); 
 if(!arr){
    fclose(f);
    return false;
}
 for(int i=0;i<p;++i){
    int id,N; 
    if(fscanf(f,"%d",&id)!=1||fscanf(f,"%d",&N)!=1){
        fclose(f);
        return false;
    }
  arr[i].id=id; 
  arr[i].N=N; 
  arr[i].a=(int*)malloc((size_t)N*N*sizeof(int)); 
  if(!arr[i].a){
    fclose(f);
    return false;
}
  if(!read_matrix(f,N,arr[i].a)){
    fprintf(stderr,"Failed to read picture matrix\n");
    fclose(f);
    return false;
}}
 int m; 
 if(fscanf(f,"%d",&m)!=1){
    fprintf(stderr,"Failed to read number of objects\n");
    fclose(f);
    return false;
}
 ObjectT* a2=(ObjectT*)calloc(m,sizeof(ObjectT)); 
 if(!a2){
    fclose(f);
    return false;
}
 for(int j=0;j<m;++j){
    int id,n; 
    if(fscanf(f,"%d",&id)!=1||fscanf(f,"%d",&n)!=1){
        fclose(f);
        return false;
    }
  a2[j].id=id; 
  a2[j].n=n; 
  a2[j].a=(int*)malloc((size_t)n*n*sizeof(int)); 
  if(!a2[j].a){
    fclose(f);
    return false;
}
  if(!read_matrix(f,n,a2[j].a)){
    fprintf(stderr,"Failed to read object matrix\n");
    fclose(f);
    return false;
}}
 fclose(f); 
 *pics=arr; 
 *P=p; 
 *objs=a2; 
 *M=m; 
 return true;
}

// This function writes the final results to an output file in a human-readable format. 
// It goes through each picture result one by one. If a match was found, it writes a line saying 
// which picture found which object at what position. If no match was found, it writes that no 
// objects were found for that picture. The output format is simple text that's easy to read 
// and understand. Returns true if writing succeeds, false if the file can't be opened or written.
bool write_output(const char* path,const MatchResult* r,int P){
 FILE* f=fopen(path,"w"); 
 if(!f){
    fprintf(stderr,"Failed to open output file: %s\n",path);
    return false;
}
 for(int i=0;i<P;++i){ 
    if(r[i].found) 
    fprintf(f,"Picture %d found Object %d in Position(%d,%d)\n",r[i].pictureId,r[i].objectId,r[i].posI,r[i].posJ);
  else fprintf(f,"Picture %d No Objects were found\n",r[i].pictureId); 
}
 fclose(f); 
 return true;
}
