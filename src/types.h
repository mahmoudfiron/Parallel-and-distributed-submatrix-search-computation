#pragma once
typedef struct{
    int id;
    int N;
    int* a;
} 
Picture;
typedef struct{
    int id;
    int n;
    int* a;
} 
ObjectT;
typedef struct{
    int pictureId;
    int found;
    int objectId;
    int posI;
    int posJ;
} 
MatchResult;
