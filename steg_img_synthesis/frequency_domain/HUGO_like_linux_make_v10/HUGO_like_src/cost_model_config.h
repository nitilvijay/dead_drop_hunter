#ifndef CONFIG_H_
#define CONFIG_H_

#include "base_cost_model_config.h"

class cost_model_config : public base_cost_model_config
{
public:
	float gamma;
	float sigma;

	cost_model_config(float payload, bool verbose, float gamma, float sigma, unsigned int stc_constr_height, int randSeed);
	~cost_model_config();
};
#endif