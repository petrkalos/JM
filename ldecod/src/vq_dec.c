#include <stdio.h>
#include <stdlib.h>
#include <emmintrin.h>
#include <float.h>
#include <math.h>

#include "global.h"
#include "mbuffer.h"
#include "memalloc.h"

#include "vq_dec.h"

float **cb[3];

float *temp = NULL;

int dim;
int dims;
int cblen;
int vqlen;

int *vqindex = NULL;

const int pos[3] = {1,17,21};
const int mask[4] = {0x01,0x02,0x04,0x08};

#define IND_SIZE 25

#define round(r) (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5)

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

static inline int is_intra(Macroblock *curr_MB)
{
	return ((curr_MB)->mb_type==SI4MB || (curr_MB)->mb_type==I4MB || (curr_MB)->mb_type==I16MB || (curr_MB)->mb_type==I8MB || (curr_MB)->mb_type==IPCM);
}

static inline int is_p(Macroblock *curr_MB){
	if((curr_MB)->mb_type==0 || (curr_MB)->mb_type==1 || (curr_MB)->mb_type==2 || (curr_MB)->mb_type==3 || (curr_MB)->mb_type==8) return 1;
	return 0;
}

void check_file(FILE *fp){
	if(fp==NULL){
		printf("Error opening file\n");
		exit(1);
	}

	if(ferror(fp)!=0){
		printf("File didn't opened correctly %d\n\n\n",strerror(ferror(fp)));
		exit(1);
	}
}

int reverse_shift(int x){
	return 64*x-32;
}

void read_vqindices(int frame){
	FILE *fpIndex;
	
	if(vqindex==NULL) return;

	fpIndex = fopen("vqindex.bin","rb");
	check_file(fpIndex);
	
	fseek(fpIndex,vqlen*frame*sizeof(int),SEEK_SET);
	fread(vqindex,sizeof(int),vqlen,fpIndex);

	fclose(fpIndex);
}

void init_codebooks(VideoParameters *vp){
	int i,pl,size,mode;
	InputParameters *Inp;
	FILE *fpYI,*fpYB,*fpYP,*fpUVI,*fpUVB,*fpUVP;

	if(temp!=NULL) return;

	Inp = vp->p_Inp;

	cblen = Inp->cblen;
	dim = Inp->dim;
	dims = (int)sqrt((double)dim);
	size = cblen*dim;


	temp = (float *)_aligned_malloc(sizeof(float)*dim,16);

	for(mode=0;mode<3;mode++){
		cb[mode] = (float **)malloc(sizeof(float *)*2);
	}

	for(mode=0;mode<3;mode++){
		for(pl=0;pl<2;pl++){
			cb[mode][pl] = (float *)_aligned_malloc(size*sizeof(float),16);
		}
	}

	fpYI = fopen(Inp->cbYI,"rb");
	fpYP = fopen(Inp->cbYP,"rb");
	fpYB = fopen(Inp->cbYB,"rb");
	fpUVI = fopen(Inp->cbUVI,"rb");
	fpUVP = fopen(Inp->cbUVP,"rb");
	fpUVB = fopen(Inp->cbUVB,"rb");

	check_file(fpYI);check_file(fpYB);check_file(fpYP);check_file(fpUVI);check_file(fpUVB);check_file(fpUVP);


	fread(cb[0][0],sizeof(float),size,fpYI);
	fread(cb[0][1],sizeof(float),size,fpUVI);
	fread(cb[1][0],sizeof(float),size,fpYP);
	fread(cb[1][1],sizeof(float),size,fpUVP);
	fread(cb[2][0],sizeof(float),size,fpYB);
	fread(cb[2][1],sizeof(float),size,fpUVB);

	fclose(fpYI);fclose(fpYB);fclose(fpYP);fclose(fpUVI);fclose(fpUVB);fclose(fpUVP);

	for(mode=0;mode<3;mode++){
		for(pl=0;pl<2;pl++){
			for(i=0;i<cblen*dim;i++){
				cb[mode][pl][i] = reverse_shift((int)round(cb[mode][pl][i]));
			}
		}
	}

	vqlen = IND_SIZE*vp->FrameSizeInMbs;
	vqindex = (int *)malloc(sizeof(int)*vqlen);

	read_vqindices(0);
}



void quantize_mb(int **mb_rres,int width, int height, int mb_y,int mb_x,int pl,Macroblock *currMB){
	int i,j,vi,vj,uv,mode=0;
	int t,idx,idx8x8;
	int addr;
	
	addr = currMB->mbAddrX;

	if(pl==2){
		uv = 2;
		pl = 1;
	}else{
		uv = pl;
	}
	
	for (i = 0; i < height/(pl+1); i+=dims){
		for(j = 0; j< width/(pl+1); j+=dims){

			t = pos[uv]+(mb_y/dims+i/dims)*4/(pl+1)+(mb_x/dims+j/dims);
			idx = vqindex[addr*IND_SIZE+t]*dim;
			idx8x8 = (mb_y/8+i/8)*2/(pl+1)+(mb_x/8+j/8);

			if(idx!=-dim && (currMB->cbp & mask[idx8x8] | pl)){
					
				if(is_intra(currMB)) mode = 0;
				else if(is_p(currMB) && currMB->b8pdir[idx8x8]==BI_PRED) mode = 1;
				else mode = 2;

				for(vi=0;vi<dims;vi++){
					for(vj=0;vj<dims;vj++){
						mb_rres[i+vi][mb_x+j+vj] = (cb[mode][pl][idx+vi*dims+vj]);
					}
				}

			}
		}
	}
}