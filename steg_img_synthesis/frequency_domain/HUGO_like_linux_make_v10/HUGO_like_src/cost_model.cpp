#include "base_cost_model.h"
#include "cost_model.h"
#include "mat2D.h"
#include "cost_model_config.h"
#include "base_cost_model_config.h"
#include <float.h>

cost_model::cost_model(mat2D<int>* cover, cost_model_config* config) : base_cost_model(cover, (base_cost_model_config *)config)
{
	this->config = config;

	mat2D<int>* cover_padded = mat2D<int>::Padding_Mirror(cover, 3, 3);
	for (int r=0; r<cover->rows; r++)
	{
		for (int c=0; c<cover->cols; c++)
		{
			// pixel_costs[0] is the cost of -1, pixel_costs[0] is the cost of no change, pixel_costs[0] is the cost of +1
			float* pixel_costs = costs + ((c+r*cover->cols)*3); 
			pixel_costs[0] = 0; pixel_costs[1] = 0; pixel_costs[2] = 0;
			calc_costs(r+3, c+3, cover_padded, pixel_costs);
		}
	}
}

cost_model::~cost_model()
{
}

void cost_model::calc_costs(int r, int c, mat2D<int>* cover_padded, float* pixel_costs) {

    eval_direction(r, c,-1, 1, cover_padded, pixel_costs); // NE
    eval_direction(r, c, 0, 1, cover_padded, pixel_costs); // E
    eval_direction(r, c, 1, 1, cover_padded, pixel_costs); // SE
    eval_direction(r, c, 1, 0, cover_padded, pixel_costs); // S
}
float cost_model::eval_cost(int k, int l, int m) {
	return pow(this->config->sigma + sqrt((float)(k*k + l*l + m*m)), -this->config->gamma);
}

void cost_model::eval_direction(int r, int c, int dir_r, int dir_c, mat2D<int>* cover_padded, float *pixel_costs) 
{
    int p[7] = { 0, 0, 0, 0, 0, 0, 0 };
    int d[6] = { 0, 0, 0, 0, 0, 0 };
    float cover_cost, stego_cost;

    for ( int k = -3; k <= 3; k++ )
		p[3 + k] = cover_padded->Read(r + dir_r*k, c + dir_c*k);
    for ( int k = 0; k < 6; k++ ) // calculate differences
        d[k] = p[k + 1] - p[k];

	// leftmost clique
    cover_cost = eval_cost( d[0], d[1], d[2] );
    stego_cost = eval_cost( d[0], d[1], d[2] - 1 ); // evaluate change by -1 => diff in stego is smaller by 1
    pixel_costs[0] += cover_cost + stego_cost;
    stego_cost = eval_cost( d[0], d[1], d[2] + 1 ); // evaluate change by +1 => diff in stego is larger by 1
    pixel_costs[2] += cover_cost + stego_cost;

    // left center clique
    cover_cost = eval_cost( d[1], d[2], d[3] );
    stego_cost = eval_cost( d[1], d[2] - 1, d[3] + 1 ); // evaluate change by -1
    pixel_costs[0] += cover_cost + stego_cost;
    stego_cost = eval_cost( d[1], d[2] + 1, d[3] - 1 ); // evaluate change by +1
    pixel_costs[2] += cover_cost + stego_cost;

    // right center clique
    cover_cost = eval_cost( d[2], d[3], d[4] );
    stego_cost = eval_cost( d[2] - 1, d[3] + 1, d[4] ); // evaluate change by -1
    pixel_costs[0] += cover_cost + stego_cost;
    stego_cost = eval_cost( d[2] + 1, d[3] - 1, d[4] ); // evaluate change by +1
    pixel_costs[2] += cover_cost + stego_cost;
   
	// rightmost clique
    cover_cost = eval_cost( d[3], d[4], d[5] );
    stego_cost = eval_cost( d[3] + 1, d[4], d[5] ); // evaluate change by -1
    pixel_costs[0] += cover_cost + stego_cost;
    stego_cost = eval_cost( d[3] - 1, d[4], d[5] ); // evaluate change by +1
    pixel_costs[2] += cover_cost + stego_cost;

    if ( p[3] == 255 ) pixel_costs[2] = FLT_MAX; // stay within dynamic range [0,255]
    if ( p[3] == 0 ) pixel_costs[0] = FLT_MAX;
    pixel_costs[1] = 0;
}
