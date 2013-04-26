#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <float.h>
#include <math.h>

#include "rdopt.h"
#include "fastnn.h"
#include "global.h"
#include "mbuffer.h"
#include "memalloc.h"

#define round(r) (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5)

float **cbI;
float **cbB;
float **cbP;

struct node *rootI[2];
struct node *rootP[2];
struct node *rootB[2];
struct context *storI[2];
struct context *storP[2];
struct context *storB[2];

float *temp;

int dim;
int dims;
int cblen;

float min_dist[2];

#define FASTNN

void check_file(FILE *fp){
	if(fp==NULL){
		printf("file open error\n\n\n");
		exit(1);
	}
}

float distance2_sse2(float *vector1, float *vector2, int dim)
{
	int i;
	float sum;
	__m128 xmm0,xmm1,xmm2;

	xmm0 = _mm_setzero_ps();
	for(i=dim;i>0;i-=4)
	{
		xmm1 = _mm_load_ps ((const float *) (vector1+0));
		xmm2 = _mm_load_ps ((const float *) (vector2+0));
		xmm1 = _mm_sub_ps (xmm1, xmm2);
		xmm1 = _mm_mul_ps (xmm1, xmm1);
		xmm0 = _mm_add_ps (xmm0, xmm1);
		vector1 += 4;
		vector2 += 4;
	}
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

int find_min(float *cb,float *vector){
	int i,min_ind;
	float min_dist,dist;
	min_dist = FLT_MAX;
	min_ind = -1;

	for(i=0;i<cblen;i++){
		dist = distance2_sse2(&cb[i*dim],vector,dim);
		if(dist<min_dist){
			min_ind = i;
			min_dist = dist;
		}
	}

	//printf("Distance %lf\n",sqrt(min_dist)/dim);
	return min_ind;
}

int reverse_shift(int x){
	return 64*x-32;
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

	//get_mem3Dint(&vp->vqIndex,3,MB_BLOCK_SIZE/dims,MB_BLOCK_SIZE/dims);
	fclose(fopen("vqindex.txt","w"));
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
#ifdef FASTNN
	initNN(&rootI[0],&storI[0],dim,cblen,cbI[0]);
	initNN(&rootI[1],&storI[1],dim,cblen,cbI[1]);
	initNN(&rootP[0],&storP[0],dim,cblen,cbP[0]);
	initNN(&rootP[1],&storP[1],dim,cblen,cbP[1]);
	initNN(&rootB[0],&storB[0],dim,cblen,cbB[0]);
	initNN(&rootB[1],&storB[1],dim,cblen,cbB[1]);	
#endif
}

void quantize_mb(int **mb_rres,int width, int height, int mb_y,int mb_x,int pl,Macroblock *currMB){
	int i,j,vi,vj,min,uv=0;
	int mode;
	float subb=0.0;

	if(pl != 0){
		uv = pl;
		pl = 1;
	}

	for (i = 0; i < height; i+=dims){
		for(j = 0; j< width; j+=dims){

			for(vi=0;vi<dims;vi++){
				for(vj=0;vj<dims;vj++){
					temp[vi*dims+vj] = (float)rshift_rnd_sf(mb_rres[i+vi][mb_x+j+vj],6);
				}
			}


			if(is_intra(currMB)) mode = 0;
			else if(is_p(currMB) && currMB->b8x8[(int)subb].pdir==BI_PRED) mode = 1;
			else mode = 2;

			if(mode==0){
#ifdef FASTNN
				min = fastNN(temp,rootI[pl],cbI[pl],dim,min_dist);
#else
				min = find_min(cbI[pl],temp);
#endif
				for(vi=0;vi<dims;vi++){
					for(vj=0;vj<dims;vj++){
						mb_rres[i+vi][mb_x+j+vj] = reverse_shift((int)(cbI[pl%2][min*dim+vi*dims+vj]));
					}
				}
				currMB->vqIndex[uv][mb_y/dims+i/dims][mb_x/dims+j/dims] = min;

			}else if(mode==1){
#ifdef FASTNN
				min = fastNN(temp,rootI[pl],cbI[pl],dim,min_dist);
#else
				min = find_min(cbI[pl],temp);
#endif
				for(vi=0;vi<dims;vi++){
					for(vj=0;vj<dims;vj++){
						mb_rres[i+vi][mb_x+j+vj] = reverse_shift((int)(cbB[pl][min*dim+vi*dims+vj]));
					}
				}
				currMB->vqIndex[uv][mb_y/dims+i/dims][mb_x/dims+j/dims] = min;
			}else if(mode==2){
#ifdef FASTNN
				min = fastNN(temp,rootI[pl],cbI[pl],dim,min_dist);
#else
				min = find_min(cbI[pl],temp);
#endif
				for(vi=0;vi<dims;vi++){
					for(vj=0;vj<dims;vj++){
						mb_rres[i+vi][mb_x+j+vj] = reverse_shift((int)(cbP[pl][min*dim+vi*dims+vj]));
					}
				}
				currMB->vqIndex[uv][mb_y/dims+i/dims][mb_x/dims+j/dims] = min;
			}

			subb+=0.5;
		}
	}
}

void write_vq(Macroblock *currMB){

	FILE *fp;
	int pl,i,j,mode;
	float subb;

	fp = fopen("vqindex.txt","a");
	fprintf(fp,"%d\n",currMB->mbAddrX);
	for(pl=0;pl<3;pl++){
		subb=0.0;
		for(i=0;i<4;i++){
			for(j=0;j<4;j++){
				if(is_intra(currMB)) mode = 0;
				else if(is_p(currMB) && currMB->b8x8[(int)subb].pdir==BI_PRED) mode = 1;
				else mode = 2;

				if(pl==0){
					fprintf(fp,"%d@%5d ",mode,currMB->p_Slice->p_RDO->vqIndex[pl][i][j]);
				}else if(i<2 && j<2){
					fprintf(fp,"%d@%5d ",mode,currMB->p_Slice->p_RDO->vqIndex[pl][i][j]);

				}
				subb+=0.5;
			}
		}
		fprintf(fp,"\n");
	}


	fclose(fp);
}