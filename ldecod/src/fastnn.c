#include <stdlib.h>
#include <stdio.h>
#include <emmintrin.h>
#include <math.h>
#include <string.h>

#include "fastnn.h"

#define ALIGNMENT 16
#define ALIGN(x,y) (((x)+(y)-1)/(y))*(y)
#define MY_DOUBLE float
#define MY_SHORT MY_DOUBLE
#define FAST_NN_THRESHOLD 3

#define my_malloc(size,d) _aligned_malloc(size,d)
#define my_free(ptr) _aligned_free(ptr)
#define distance2(v1,v2,dim) distance2_sse2_2(v1,v2,dim)
#define my_distance2(v1,v2,dim,min_dist) distance2(v1,v2,dim)
#define signed_distance(v,h,c0,dim) signed_distance_sse2(v,h,c0,dim)
#define calc_hyperplane(v1,v2,h,dim) calc_hyperplane_sse2(v1,v2,h,dim)
#define accumulate_vector(v1,v2,dim) accumulate_vector_sse2(v1,v2,dim)

__inline MY_DOUBLE distance2_sse2_2(MY_DOUBLE *vector1, MY_DOUBLE *vector2, int dim)
{
	int i;
	MY_DOUBLE sum;
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

MY_DOUBLE calc_hyperplane_sse2(MY_DOUBLE *vector1, MY_DOUBLE *vector2, MY_DOUBLE *hyperplane, int dim)
{
	int i;
	MY_DOUBLE val;
	MY_DOUBLE *ptr;
	__m128 xmm0,xmm1,xmm2,xmm3,xmm4;

	ptr = hyperplane;
	xmm0 = _mm_setzero_ps();
	xmm1 = _mm_setzero_ps();
	for(i=dim;i>0;i-=4)
	{
		xmm3 = _mm_load_ps ((const float *) (vector2));
		xmm2 = _mm_load_ps ((const float *) (vector1));
		xmm4 = _mm_sub_ps (xmm3, xmm2);
		xmm3 = _mm_mul_ps (xmm3, xmm3);
		_mm_store_ps ((float *) ptr,xmm4);
		xmm2 = _mm_mul_ps (xmm2, xmm2);
		xmm4 = _mm_mul_ps (xmm4, xmm4);
		xmm1 = _mm_add_ps (xmm1, xmm2);
		xmm0 = _mm_add_ps (xmm0, xmm4);
		xmm1 = _mm_sub_ps (xmm1, xmm3);
		vector2 += 4;
		vector1 += 4;
		ptr += 4;
	}
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0x4E); // 1 0 3 2
	xmm0 = _mm_add_ps (xmm0, xmm2); // (3+1) (2+0) (3+1) (2+0)
	xmm2 = _mm_shuffle_ps (xmm0, xmm0, 0xB1); // 2 3 0 1
	xmm0 = _mm_add_ps (xmm0, xmm2);
	xmm0 = _mm_rsqrt_ps (xmm0);
	for(i=dim;i>0;i-=4)
	{
		xmm2 = _mm_load_ps ((const float *) (hyperplane));
		xmm2 = _mm_mul_ps (xmm2, xmm0);
		_mm_store_ps ((float *) hyperplane,xmm2);
		hyperplane += 4;
	}
	xmm2 = _mm_shuffle_ps (xmm1, xmm1, 0xEE);
	xmm1 = _mm_add_ps (xmm1, xmm2);
	xmm2 = _mm_shuffle_ps (xmm1, xmm1, 0x55);
	xmm1 = _mm_add_ps (xmm1, xmm2);
	xmm0 = _mm_mul_ps (xmm0, xmm1);
	_mm_store_ss (&val, xmm0);
	return val*(MY_DOUBLE) 0.5;
}

void accumulate_vector_sse2(double *vector1, MY_DOUBLE *vector2, int dim)
{
	int i;
	__m128d xmm2, xmm3, xmm4, xmm5;
	__m128 xmm1;

	for(i=dim;i>0;i-=4)
	{
		xmm1 = _mm_load_ps ((const float *) (vector2+0));
		xmm2 = _mm_load_pd ((const double *) (vector1+0));
		xmm3 = _mm_load_pd ((const double *) (vector1+2));
		xmm4 = _mm_cvtps_pd (xmm1);
		xmm1 = _mm_shuffle_ps (xmm1, xmm1, 0xEE);
		xmm5 = _mm_cvtps_pd (xmm1);
		xmm2 = _mm_add_pd (xmm2, xmm4);
		xmm3 = _mm_add_pd (xmm3, xmm5);
		_mm_store_pd ((double *) (vector1+0), xmm2);
		_mm_store_pd ((double *) (vector1+2), xmm3);
		vector1 += 4;
		vector2 += 4;
	}
}

struct node *allocate_node(int num_of_clusters,int dim)
{
	struct node *root = NULL;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(dim>0 && num_of_clusters>=1)
	{
		root = (struct node *) my_malloc(1*sizeof(struct node),ALIGNMENT);
		if(root==NULL)
		{
			printf("Not enough memory (%d bytes) for %s - exiting\n",
				1*sizeof(struct node), "root");
			exit(-1);
		}
		root->pairs = (struct paired *) my_malloc(num_of_clusters*sizeof(struct paired),ALIGNMENT);
		if(root->pairs==NULL)
		{
			printf("Not enough memory (%d bytes) for %s - exiting\n",
				num_of_clusters*sizeof(struct paired), "root->pairs");
			exit(-1);
		}
		if(num_of_clusters>FAST_NN_THRESHOLD)
		{
			root->hyperplane = (MY_DOUBLE *) my_malloc(dim2*sizeof(MY_DOUBLE),ALIGNMENT);
			if(root->hyperplane==NULL)
			{
				printf("Not enough memory (%d bytes) for %s - exiting\n",
					dim2*sizeof(MY_DOUBLE), "root->hyperplane");
				exit(-1);
			}
		}
		else
			root->hyperplane = NULL;
		root->left = root->right = NULL;
	}
	return root;
}

void free_node(struct node *root)
{
	if(root!=NULL)
	{
		if(root->pairs!=NULL)
			my_free(root->pairs);
		if(root->hyperplane!=NULL)
			my_free(root->hyperplane);
		my_free(root);
	}
}

void free_tree(struct node *root)
{
	if(root!=NULL)
	{
		free_tree(root->left);
		free_tree(root->right);
		free_node(root);
	}
}

MY_DOUBLE signed_distance_sse2(MY_DOUBLE *vector, MY_DOUBLE *hyperplane, MY_DOUBLE c0, int dim)
{
	int i;
	MY_DOUBLE sum;
	__m128 xmm0,xmm1,xmm2;

	xmm0 =  _mm_load_ss((float *) &c0);
	for(i=dim;i>0;i-=4)
	{
		xmm1 = _mm_load_ps ((const float *) (vector));
		xmm2 = _mm_load_ps ((const float *) (hyperplane));
		xmm1 = _mm_mul_ps (xmm1, xmm2);
		xmm0 = _mm_add_ps (xmm0, xmm1);
		vector += 4;
		hyperplane += 4;
	}
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0xEE);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	xmm1 = _mm_shuffle_ps (xmm0, xmm0, 0x55);
	xmm0 = _mm_add_ps (xmm0, xmm1);
	_mm_store_ss (&sum, xmm0);
	return sum;
}

double kmeans_initialize2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map)
{
	MY_DOUBLE *clusters = storage->clusters2;
	int *indices = storage->indices2;
	MY_DOUBLE *distances = storage->distances2;
	int *cluster_count = storage->cluster_count2;
	double *new_clusters = storage->new_clusters2;
	int *diff_count = &storage->diff_count2;
	int i;
	MY_DOUBLE *p1;
	MY_DOUBLE *p2;
	MY_DOUBLE dist;
	MY_DOUBLE min_dist;
	MY_DOUBLE max_dist;
	int max_ind;
	double total_sum;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	*diff_count = num_of_vectors;
	if(num_of_vectors<2)
	{
		return -1.0;
	}
	memset(new_clusters,0,dim2*sizeof(double));
	memset(cluster_count,0,2*sizeof(int));
	memset(indices,0,num_of_vectors*sizeof(int));
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[map[i].index*dim2];
		accumulate_vector(new_clusters,p1,dim);
	}
	cluster_count[0] = num_of_vectors;

	for(i=0;i<dim;i++)
	{
		clusters[i] = (MY_DOUBLE) new_clusters[i]/(MY_DOUBLE) num_of_vectors;
	}
	for(;i<dim2;i++)
	{
		clusters[i] = (MY_DOUBLE) 0;
	}
	// clusters[0] is the overall centroid
	max_dist = -1.0;
	for(i=0;i<num_of_vectors;i++)
	{
		min_dist = distance2(&training[map[i].index*dim2],clusters,dim);
		if(min_dist>max_dist)
		{
			max_dist = min_dist;
			max_ind = i;
		}
	}
	p1 = &training[map[max_ind].index*dim2];
	for(i=0;i<dim;i++)
	{
		clusters[i] = p1[i];
	}
	// clusters[0] is the first centroid
	max_dist = -1.0;
	for(i=0;i<num_of_vectors;i++)
	{
		min_dist = distance2(&training[map[i].index*dim2],clusters,dim);
		distances[i] = min_dist;
		if(min_dist>max_dist)
		{
			max_dist = min_dist;
			max_ind = i;
		}
	}
	p2 = &clusters[dim2];
	p1 = &training[map[max_ind].index*dim2];
	for(i=0;i<dim;i++)
	{
		p2[i] = p1[i];
	}
	for(;i<dim2;i++)
	{
		p2[i] = (MY_DOUBLE) 0;
	}
	// p2=clusters[1] is the second centroid
	total_sum = 0.0;
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[map[i].index*dim2];
		min_dist = distances[i];
		dist = my_distance2(p1,p2,dim,min_dist);
		if(dist<min_dist)
		{
			min_dist = dist;
			distances[i] = dist;
			indices[i] = 1;
		}
		total_sum += min_dist;
	}
	return ((double) total_sum)/((double) num_of_vectors);
}

void kmeans_cluster2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map)
{
	MY_DOUBLE *clusters = storage->clusters2;
	int *indices = storage->indices2;
	MY_DOUBLE *distances = storage->distances2;
	int *cluster_count = storage->cluster_count2;
	double *new_clusters = storage->new_clusters2;
	int *diff_count = &storage->diff_count2;
	int i, j;
	MY_DOUBLE *p1;
	int min_ind;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(num_of_vectors<2)
	{
		return;
	}
	memset(new_clusters,0,2*dim2*sizeof(double));
	memset(cluster_count,0,2*sizeof(int));
	for(i=0;i<num_of_vectors;i++)
	{
		min_ind = indices[i];
		accumulate_vector(&new_clusters[min_ind*dim2],&training[map[i].index*dim2],dim);
		cluster_count[min_ind]++;
	}
	for(j=0;j<2;j++)
	{
		if((min_ind=cluster_count[j])>0)
		{
			double *p2;
			p1 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];
			for(i=0;i<dim;i++)
			{
				p1[i] = (MY_DOUBLE) p2[i]/(MY_DOUBLE) min_ind;
			}
		}
	}
}

double kmeans_iterate2_map(struct context *storage, MY_DOUBLE *training, int dim, int num_of_vectors, struct paired *map)
{
	MY_DOUBLE *clusters = storage->clusters2;
	int *indices = storage->indices2;
	MY_DOUBLE *distances = storage->distances2;
	int *cluster_count = storage->cluster_count2;
	double *new_clusters = storage->new_clusters2;
	int *diff_count = &storage->diff_count2;
	int i, j;
	MY_DOUBLE *p1;
	MY_DOUBLE dist;
	MY_DOUBLE min_dist;
	int min_ind;
	double total_sum;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(num_of_vectors<2)
	{
		*diff_count = -1;
		return -1.0;
	}
	memset(new_clusters,0,2*dim2*sizeof(double));
	memset(cluster_count,0,2*sizeof(int));
	*diff_count = 0;
	total_sum = 0.0;
	for(i=0;i<num_of_vectors;i++)
	{
		p1 = &training[map[i].index*dim2];
		min_ind = indices[i];
		min_dist = distance2(p1,&clusters[min_ind*dim2],dim);
		dist = my_distance2(p1,&clusters[(1-min_ind)*dim2],dim,min_dist);
		if(dist<min_dist)
		{
			min_dist = dist;
			min_ind = 1-min_ind;
			(*diff_count)++;
			indices[i] = min_ind;
		}
		accumulate_vector(&new_clusters[min_ind*dim2],p1,dim);
		cluster_count[min_ind]++;
		distances[i] = min_dist;
		total_sum += min_dist;
	}
	for(j=0;j<2;j++)
	{
		if((min_ind=cluster_count[j])>0)
		{
			double *p2;
			MY_DOUBLE *p3;
			p3 = &clusters[j*dim2];
			p2 = &new_clusters[j*dim2];

			for(i=0;i<dim;i++)
			{
				p3[i] = (MY_DOUBLE) p2[i]/(MY_DOUBLE) min_ind;
			}
		}
	}
	return ((double) total_sum)/((double) (num_of_vectors*dim));
}

int increasing(const void *iptr1, const void *iptr2)
{
	struct paired *ptr1 = (struct paired *) iptr1;
	struct paired *ptr2 = (struct paired *) iptr2;

	if(ptr1->signed_distance<ptr2->signed_distance)
		return -1;
	else
		return 1;
}

int binary_search(struct paired *pairs, int count, MY_DOUBLE mid_point)
{
	int min_i, max_i, mid_i;
	min_i = 0;
	max_i = count-1;
	for(;max_i>min_i+1;)
	{
		mid_i = (max_i+min_i+1)/2;
		if(pairs[mid_i].signed_distance<=mid_point)
			min_i = mid_i;
		else
			max_i = mid_i;
	}
	return max_i;
}

void tree_structure(struct node *root, MY_DOUBLE *clusters,int dim,struct context *storage)
{
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(root->count>FAST_NN_THRESHOLD)
	{
		int i;
		int iter = 0;
		double ret_val;
		int mid_i;
		ret_val = kmeans_initialize2_map(storage, clusters, dim, root->count, root->pairs);
		kmeans_cluster2_map(storage, clusters, dim, root->count, root->pairs);
		//printf("TREE: iter=%3d, ret_val= %lf, diff=%7d, H=%lf\n",iter,ret_val,diff_count2,entropy(cluster_count2,2)/((double) dim));
		iter++;
		for(iter=1;iter<100 && storage->diff_count2>0;iter++)
		{
			ret_val = kmeans_iterate2_map(storage, clusters, dim,root->count, root->pairs);
			//printf("TREE: iter=%3d, ret_val= %lf, diff=%7d, H=%lf\n",iter,ret_val,diff_count2,entropy(cluster_count2,2)/((double) dim));
		}
		root->c0 = calc_hyperplane(&storage->clusters2[0*dim2],&storage->clusters2[1*dim2],root->hyperplane,dim);
		for(i=0;i<root->count;i++)
		{
			root->pairs[i].signed_distance = signed_distance(&clusters[root->pairs[i].index*dim2],root->hyperplane,root->c0,dim);
		}
		qsort((void *) root->pairs,root->count,sizeof(struct paired),increasing);
		mid_i = binary_search(root->pairs,root->count,0.0);
		root->left = allocate_node(mid_i,dim);
		root->left->count = mid_i;
		for(i=0;i<mid_i;i++)
		{
			root->left->pairs[i].index = root->pairs[i].index;
		}
		root->right = allocate_node(root->count-mid_i,dim);
		root->right->count = root->count-mid_i;
		for(i=mid_i;i<root->count;i++)
		{
			root->right->pairs[i-mid_i].index = root->pairs[i].index;
		}
		tree_structure(root->left,clusters,dim,storage);
		tree_structure(root->right,clusters,dim,storage);
	}
}

int fastNN(MY_DOUBLE *vector, struct node *root, MY_DOUBLE *clusters, int dim, MY_DOUBLE *min_dist2)
{
	int i;
	int min_ind;
	MY_DOUBLE dist;
	MY_DOUBLE test_dist;
	int dim2 = ALIGN(dim,ALIGNMENT/sizeof(MY_SHORT));

	if(root->count>FAST_NN_THRESHOLD)
	{
		double limit;
		struct paired *ptr;
		dist = signed_distance(vector,root->hyperplane,root->c0,dim);
		if(dist<=0.0)
		{
			min_ind = fastNN(vector,root->left,clusters,dim,min_dist2);
			limit = min_dist2[1]+dist;
			i = root->left->count;
			for(ptr = &root->pairs[i];i<root->count && ptr->signed_distance<limit;i++,ptr++)
			{
				test_dist = my_distance2(vector,&clusters[ptr->index*dim2],dim,min_dist2[0]);
				if(test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE) sqrt(test_dist);
					limit = min_dist2[1]+dist;
					min_ind = ptr->index;
				}
			}
		}
		else
		{
			min_ind = fastNN(vector,root->right,clusters,dim,min_dist2);
			limit = dist-min_dist2[1];
			i = root->left->count-1;
			for(ptr = &root->pairs[i];i>=0 && ptr->signed_distance>limit;i--,ptr--)
			{
				test_dist = my_distance2(vector,&clusters[ptr->index*dim2],dim,min_dist2[0]);
				if(test_dist<min_dist2[0])
				{
					min_dist2[0] = test_dist;
					min_dist2[1] = (MY_DOUBLE) sqrt(test_dist);
					limit = dist-min_dist2[1];
					min_ind = ptr->index;
				}
			}
		}
	}
	else
	{
		min_dist2[0] = distance2(vector,&clusters[root->pairs[0].index*dim2],dim);
		min_ind = root->pairs[0].index;
#if FAST_NN_THRESHOLD==2
		if(root->count>1)
		{
			test_dist = my_distance2(vector,&clusters[root->pairs[1].index*dim2],dim,min_dist2[0]);
#ifdef PROFILE
			count_distance++;
#endif
			if(test_dist<min_dist2[0])
			{
				min_ind = root->pairs[1].index;
				min_dist2[0] = test_dist;
			}
		}
#elif FAST_NN_THRESHOLD>1
		for(i=1;i<root->count;i++)
		{
			test_dist = my_distance2(vector,&clusters[root->pairs[i].index*dim2],dim,min_dist2[0]);
			if(test_dist<min_dist2[0])
			{
				min_ind = root->pairs[i].index;
				min_dist2[0] = test_dist;
			}
		}
#endif
		min_dist2[1] = (MY_DOUBLE) sqrt(min_dist2[0]);
	}
	return min_ind;
}

void allocNN(struct node **root,struct context **storage,int dim2,int num_of_clusters){
	int dim = dim2;
	(*storage) = (struct context *) my_malloc(1*sizeof(struct context),ALIGNMENT);
	if((*storage)==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			1*sizeof(struct context), "(*storage)");
		exit(-1);
	}
	(*storage)->clusters2 = (MY_DOUBLE *) my_malloc(2*dim2*sizeof(MY_DOUBLE),ALIGNMENT);
	if((*storage)->clusters2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			2*dim2*sizeof(MY_DOUBLE), "(*storage)->clusters2");
		exit(-1);
	}
	(*storage)->indices2 = (int *) my_malloc(num_of_clusters*sizeof(int),ALIGNMENT);
	if((*storage)->indices2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_clusters*sizeof(MY_DOUBLE), "(*storage)->indices2");
		exit(-1);
	}
	(*storage)->cluster_count2 = (int *) my_malloc(2*sizeof(int),ALIGNMENT);
	if((*storage)->cluster_count2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			2*sizeof(int), "(*storage)->cluster_count2");
		exit(-1);
	}
	(*storage)->distances2 = (MY_DOUBLE *) my_malloc(num_of_clusters*sizeof(MY_DOUBLE),ALIGNMENT);
	if((*storage)->distances2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			num_of_clusters*sizeof(MY_DOUBLE), "(*storage)->distances2");
		exit(-1);
	}
	(*storage)->new_clusters2 = (double *) my_malloc(2*dim2*sizeof(double),ALIGNMENT);
	if((*storage)->new_clusters2==NULL)
	{
		printf("Not enough memory (%d bytes) for %s - exiting\n",
			2*dim2*sizeof(double), "new_clusters2");
		exit(-1);
	}

	{
		int i;
		(*root) = allocate_node(num_of_clusters,dim);
		(*root)->count = num_of_clusters;
		for(i=0;i<num_of_clusters;i++)
		{
			(*root)->pairs[i].index = i;
		}
	}
}

void initNN(struct node **root,struct context **storage,int dim2,int num_of_clusters,float *clusters){
	allocNN(root,storage,dim2,num_of_clusters);
	tree_structure(*root,clusters,dim2,*storage);
}