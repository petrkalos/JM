#include "global.h"

void init_codebooks(VideoParameters *vp);
void quantize_mb(int **mb_rres,int width, int height, int mb_y,int mb_x,int pl,Macroblock *currMB);