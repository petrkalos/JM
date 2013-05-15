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

int *vqindex;

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
	FILE *fpYI,*fpYB,*fpYP,*fpUVI,*fpUVB,*fpUVP,*fp_Index;

	Inp = vp->p_Inp;

	cblen = Inp->cblen;
	dim = Inp->dim;
	dims = (int)sqrt((double)dim);
	size = cblen*dim;

	temp = (float *)_aligned_malloc(sizeof(float)*dim,16);

	vqindex = (int *)malloc(sizeof(int *)*1350*21);

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

	fp_Index = fopen("vqindex.bin","rb");
	if(fp_Index==NULL){
		printf("Error opening vq indices \n");
		exit(1);
	}
	fread(vqindex,sizeof(int),1350*25,fp_Index);

	fclose(fp_Index);

}

int reverse_shift(int x){
	return 64*x-32;
}

float distance2_c(float *vector1, float *vector2, int dim)
{
	int i;
	double sum;
	float diff;

	sum = 0.0;
	for(i=0;i<dim;i++)
	{
		diff = vector1[i] - vector2[i];
		sum += diff*diff;
	}
	return (float) sum;
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

void quantize_mb(int **mb_rres,int width, int height, int mb_y,int mb_x,int pl,Macroblock *currMB){
	static const int pos[3] = {1,17,21};
	int i,j,vi,vj,uv;
	int addr;
	
	addr = currMB->mbAddrX;

	if(pl==2){
		uv = 2;
		pl = 1;
	}else{
		uv = pl;
	}

	if(vqindex[addr*25]==addr){
		for (i = 0; i < height/(pl+1); i+=dims){
			for(j = 0; j< width/(pl+1); j+=dims){
				if(vqindex[addr*25+pos[uv]]!=-1){
					int idx,t;
					for(vi=0;vi<dims;vi++){
						for(vj=0;vj<dims;vj++){
							temp[vi*dims+vj] = (float)rshift_rnd_sf(mb_rres[i+vi][mb_x+j+vj],6);
						}
					}

					t = pos[uv]+(mb_y/dims+i/dims)*4/(pl+1)+(mb_x/dims+j/dims);

					idx = vqindex[addr*25+t]*dim;
					//idx = find_min(cbI[pl],temp)*dim;
						
					for(vi=0;vi<dims;vi++){
						for(vj=0;vj<dims;vj++){
							mb_rres[i+vi][mb_x+j+vj] = reverse_shift(cbI[pl][idx+vi*dims+vj]);
						}
					}
				}
			}
		}
		
	}else{
		printf("Indices sync failed\n");
	}
}