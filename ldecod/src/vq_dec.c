#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <float.h>
#include <math.h>

#include "global.h"
#include "mbuffer.h"
#include "memalloc.h"

float **cbI;
float **cbB;
float **cbP;

float *temp;

int dim;
int dims;
int cblen;

FILE *fp;

#define round(r) (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5)

void check_file(FILE *fp){
	if(fp==NULL){
		printf("file open error\n\n\n");
		exit(1);
	}
}

void init_codebooks(VideoParameters *vp){
	int pl,size;
	InputParameters *Inp;
	FILE *fpYI,*fpYB,*fpYP,*fpUVI,*fpUVB,*fpUVP;

	Inp = vp->p_Inp;

	cblen = Inp->cblen;
	dim = Inp->dim;
	dims = (int)sqrt((double)dim);
	size = cblen*dim;

	temp = (float *)_aligned_malloc(sizeof(float)*dim,16);

	cbI = (float **)malloc(sizeof(float *)*2);
	cbB = (float **)malloc(sizeof(float *)*2);
	cbP = (float **)malloc(sizeof(float *)*2);

	for(pl=0;pl<2;pl++){
		cbI[pl] = (float *)_aligned_malloc(size*sizeof(float),16);
		cbB[pl] = (float *)_aligned_malloc(size*sizeof(float),16);
		cbP[pl] = (float *)_aligned_malloc(size*sizeof(float),16);
	}

	fpYI = fopen(Inp->cbYI,"rb");
	fpYP = fopen(Inp->cbYP,"rb");
	fpYB = fopen(Inp->cbYB,"rb");
	fpUVI = fopen(Inp->cbUVI,"rb");
	fpUVP = fopen(Inp->cbUVP,"rb");
	fpUVB = fopen(Inp->cbUVB,"rb");

	check_file(fpYI);check_file(fpYB);check_file(fpYP);check_file(fpUVI);check_file(fpUVB);check_file(fpUVP);

	fread(cbI[0],sizeof(float),size,fpYI);
	fread(cbI[1],sizeof(float),size,fpUVI);
	fread(cbP[0],sizeof(float),size,fpYP);
	fread(cbP[1],sizeof(float),size,fpUVP);
	fread(cbB[0],sizeof(float),size,fpYB);
	fread(cbB[1],sizeof(float),size,fpUVB);

	fclose(fpYI);fclose(fpYB);fclose(fpYP);fclose(fpUVI);fclose(fpUVB);fclose(fpUVP);

	for(pl=0;pl<cblen*dim;pl++){
		cbI[0][pl] = (float)round(cbI[0][pl]);
		cbI[1][pl] = (float)round(cbI[1][pl]);
		cbP[0][pl] = (float)round(cbP[0][pl]);
		cbP[1][pl] = (float)round(cbP[1][pl]);
		cbB[0][pl] = (float)round(cbB[0][pl]);
		cbB[1][pl] = (float)round(cbB[1][pl]);
	}

	fp = fopen("vqindex.txt","r");
}

void quantize_mb(int **mb_rres,int width, int height, int mb_y,int mb_x,int pl,Macroblock *currMB){
	static int cnt = 0;
	static int mbAddrX;
	int i,j,vi,vj,min,uv=0;
	int size;
	int mode,idx;
	float subb=0.0;

	if(currMB->mb_type == I8MB)
		size = 4;
	else
		size = 1;

	if(cnt%24==0){
		fscanf(fp,"%d",&mbAddrX);
		cnt = 0;
	}
	printf("%d - %d\n",mbAddrX,currMB->mbAddrX);

	for(i=0;i<size;i++){
		fscanf(fp,"%d@%d",&mode,&idx);
		printf("%d@%d\n",mode,idx);
	}

	cnt+=size;
	if(mbAddrX!=currMB->mbAddrX){
		printf("MB address mismatch\n");
		exit(1);
	}
}