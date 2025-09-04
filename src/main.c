#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "types.h"
#include "io.h"
#include "compute.h"
#ifdef USE_CUDA
#include "cuda_match.h"
#endif

// This helper function sends one integer from process 0 to all other processes in the MPI program. 
// All processes must call this function at the same time. Process 0 has the real value, and all 
// other processes receive a copy of that value. This ensures every process has the same integer value.
static void bcast_int(int* v){
  MPI_Bcast(v,1,MPI_INT,0,MPI_COMM_WORLD);
} 

// This helper function sends one double (decimal number) from process 0 to all other processes. 
// It works exactly like bcast_int but for decimal numbers instead of whole numbers. All processes 
// must call this at the same time to stay synchronized.
static void bcast_double(double* v){
  MPI_Bcast(v,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
}

// This is the main program that coordinates parallel pattern matching across multiple processes. 
// First, it initializes MPI and checks command line arguments. Then process 0 reads the input file 
// containing pictures and objects to search for. All processes receive copies of this data through 
// broadcasting. Each process takes turns working on different pictures using a round-robin system 
// (process 0 gets pictures 0,3,6..., process 1 gets 1,4,7..., etc). After finding matches, all 
// processes send their results back to process 0, which combines everything and writes the final 
// output file. Finally, all memory is cleaned up and MPI is shut down properly.
int main(int argc,char** argv){
  MPI_Init(&argc,&argv); 
  int rank=0,size=1; 
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); 
  MPI_Comm_size(MPI_COMM_WORLD,&size);
 if(argc<3){ 
  if(rank==0) 
  fprintf(stderr,"Usage: %s <input.txt> <output.txt>\n",argv[0]); 
MPI_Finalize(); 
return 1; 
}
 const char* inPath=argv[1]; 
 const char* outPath=argv[2];
 double threshold=0.0; 
 Picture* pics_root=NULL; 
 int P_root=0; 
 ObjectT* objs_root=NULL; 
 int M_root=0;
 Picture* pics=NULL; 
 int P=0; 
 ObjectT* objs=NULL; 
 int M=0;
 if(rank==0){ 
  if(!read_input(inPath,&threshold,&pics_root,&P_root,&objs_root,&M_root))
  {
    fprintf(stderr,"Input parsing failed.\n"); 
    MPI_Abort(MPI_COMM_WORLD,2);
  } 
  P=P_root; 
  M=M_root; 
}
if (rank == 0) {
    fprintf(stderr, "[rank %d] finished reading %s\n", rank, inPath);
}
 bcast_double(&threshold); 
 bcast_int(&P); 
 bcast_int(&M);
 pics=(Picture*)calloc(P,sizeof(Picture)); 
 objs=(ObjectT*)calloc(M,sizeof(ObjectT));
 for(int i=0;i<P;++i) {
  int id=0,N=0; 
  if(rank==0){
    id=pics_root[i].id; 
    N=pics_root[i].N;
  } 
  bcast_int(&id); 
  bcast_int(&N);
  pics[i].id=id; 
  pics[i].N=N; 

  if(rank!=0) {
    pics[i].a=(int*)malloc((size_t)N*N*sizeof(int)); 
  }

  int count=N*N; 
  int* buf=(rank==0)?pics_root[i].a:pics[i].a; 
  MPI_Bcast(buf,count,MPI_INT,0,MPI_COMM_WORLD); 
  if(rank==0) 
  pics[i].a=pics_root[i].a; 
}
 for(int j=0;j<M;++j){
  int id=0, n=0; 
  if(rank==0){
    id=objs_root[j].id; 
    n=objs_root[j].n;
  } 
  bcast_int(&id); 
  bcast_int(&n);
  objs[j].id=id; 
  objs[j].n=n; 

  if(rank!=0) {
    objs[j].a=(int*)malloc((size_t)n*n*sizeof(int)); 
  }

  int count=n*n; 
  int* buf=(rank==0)?objs_root[j].a:objs[j].a; 
  MPI_Bcast(buf,count,MPI_INT,0,MPI_COMM_WORLD); 
  if(rank==0) 
  objs[j].a=objs_root[j].a; 
}
 int local_cap=(P+size-1)/size; 
 MatchResult* local=(MatchResult*)malloc((size_t)local_cap*sizeof(MatchResult)); 
 int lc=0;
for (int idx = rank; idx < P; idx += size) {
    MatchResult r;
#ifdef USE_CUDA
    // Try GPU first. If no GPU / error / no match on GPU → fall back to CPU.
    if (!cuda_find_match_for_picture(&pics[idx], objs, M, threshold, &r)) {
        find_match_for_picture(&pics[idx], objs, M, threshold, &r);
    }
#else
    // No CUDA build: use the existing CPU/OpenMP path
    find_match_for_picture(&pics[idx], objs, M, threshold, &r);
#endif
    local[lc++] = r;
}

 if(rank==0){ 
  MatchResult* all=(MatchResult*)malloc((size_t)P*sizeof(MatchResult)); 
  int k=0; 
  for(int idx=rank; idx<P; idx+=size) 
  all[idx]=local[k++];
  for(int src=1; src<size; ++src){ 
    int count=0; 
    MPI_Recv(&count,1,MPI_INT,src,100,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
    int* buf=(int*)malloc((size_t)count*5*sizeof(int));
   MPI_Recv(buf,count*5,MPI_INT,src,101,MPI_COMM_WORLD,MPI_STATUS_IGNORE); 
   for(int t=0;t<count;++t){ 
    int pictureId=buf[t*5+0], found=buf[t*5+1], objectId=buf[t*5+2], posI=buf[t*5+3], posJ=buf[t*5+4];
    int idxPic=-1; 
    for(int i=0;i<P;++i){ 
      if(pics[i].id==pictureId){ 
        idxPic=i; 
        break; 
      } } 
      if(idxPic>=0){ 
        all[idxPic].pictureId=pictureId; 
        all[idxPic].found=found; 
        all[idxPic].objectId=objectId; 
        all[idxPic].posI=posI; 
        all[idxPic].posJ=posJ; 
      } } 
      free(buf); 
    }
    if (rank == 0) {
    fprintf(stderr, "[rank %d] writing results to %s\n", rank, outPath);
}

  write_output(outPath,all,P); 
  free(all);
 } else { 
  MPI_Send(&lc,1,MPI_INT,0,100,MPI_COMM_WORLD); 
  int* buf=(int*)malloc((size_t)lc*5*sizeof(int)); 
  for(int i=0;i<lc;++i){ 
    buf[i*5+0]=local[i].pictureId; 
    buf[i*5+1]=local[i].found; 
    buf[i*5+2]=local[i].objectId; 
    buf[i*5+3]=local[i].posI; 
    buf[i*5+4]=local[i].posJ; 
  } 
  MPI_Send(buf,lc*5,MPI_INT,0,101,MPI_COMM_WORLD); 
  free(buf); 
}
 free(local); 
 if(rank==0){ 
  for(int i=0;i<P_root;++i) 
  free(pics_root[i].a); 
  for(int j=0;j<M_root;++j) 
  free(objs_root[j].a); 
  free(pics_root); 
  free(objs_root); 
  free(pics); 
  free(objs); 
}
 else { 
  for(int i=0;i<P;++i) 
  free(pics[i].a); 
for(int j=0;j<M;++j) 
free(objs[j].a); 
free(pics); 
free(objs); 
}
 MPI_Finalize(); 
 return 0;
}
