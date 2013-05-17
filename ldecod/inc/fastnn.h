#ifndef _FASTNN_H_
#define _FASTNN_H_

#define MY_DOUBLE float
#define MY_SHORT MY_DOUBLE

struct paired {
	int index;
	MY_DOUBLE signed_distance;
};

struct node
{
	struct node *left;
	struct node *right;
	MY_DOUBLE c0;
	MY_DOUBLE *hyperplane;
	int count;
	struct paired *pairs;
};

struct context {
	MY_DOUBLE *clusters2;
	int *indices2;
	int *cluster_count2;
	MY_DOUBLE *distances2;
	double *new_clusters2;
	int diff_count2;
};

void initNN(struct node **root,struct context **storage,int dim2,int num_of_clusters,float *clusters);
int fastNN(MY_DOUBLE *vector, struct node *root, MY_DOUBLE *clusters, int dim, MY_DOUBLE *min_dist2);

#endif