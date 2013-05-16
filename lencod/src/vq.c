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
#define FASTNN
#define round(r) (r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5)


//cb[mode={I,P,B}][plane={Y,UV}][codebookdim]
float **cb[3];
float *temp;

#ifdef FASTNN
	struct node *root[3][2];
	struct context *stor[3][2];
	float min_dist[2];
#endif

int dim;
int dims;
int cblen;

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
	int i,pl,size,mode;
	InputParameters *Inp;
	FILE *fpYI,*fpYB,*fpYP,*fpUVI,*fpUVB,*fpUVP;

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

	fclose(fopen("vqindex.bin","wb"));
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

#ifdef FASTNN
	for(mode=0;mode<3;mode++){
		for(pl=0;pl<2;pl++){
			initNN(&root[mode][pl],&stor[mode][pl],dim,cblen,cb[mode][pl]);
		}
	}
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
					temp[vi*dims+vj] = (float)(mb_rres[i+vi][mb_x+j+vj]);
				}
			}

			if(is_intra(currMB)) mode = 0;
			else if(is_p(currMB) && currMB->b8x8[(int)subb].pdir==BI_PRED) mode = 1;
			else mode = 2;

#ifdef FASTNN
			min = fastNN(temp,root[mode][pl],cb[mode][pl],dim,min_dist);
#else
			min = find_min(cbI[pl],temp);
#endif

			for(vi=0;vi<dims;vi++){
				for(vj=0;vj<dims;vj++){
					mb_rres[i+vi][mb_x+j+vj] = ((int)(cb[mode][pl][min*dim+vi*dims+vj]));
				}
			}

			currMB->vqIndex[uv][mb_y/dims+i/dims][mb_x/dims+j/dims] = min;
			
			subb+=0.5;
		}
	}
}

void write_vq(Macroblock *currMB){
	FILE *fp;
	int i,pl;
	struct rdo_structure    *p_RDO;

	p_RDO = currMB->p_Slice->p_RDO;

	fp = fopen("vqindex.bin","ab");
	check_file(fp);
	fwrite(&currMB->mbAddrX,sizeof(int),1,fp);


	pl=0;
	for(i=0;i<4;i++){
		fwrite(&p_RDO->vqIndex[pl][i],sizeof(int),4,fp);
	}

	pl++;
	for(;pl<3;pl++){
		for(i=0;i<2;i++){
			fwrite(&p_RDO->vqIndex[pl][i],sizeof(int),2,fp);
		}
	}
	
	fclose(fp);
}