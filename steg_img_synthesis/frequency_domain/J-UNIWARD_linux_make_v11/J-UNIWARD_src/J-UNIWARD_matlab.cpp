#include <vector>
#include "cost_model.h"
#include "../include/mat2D.h"
#include "mi_embedder.h"
#include "cost_model_config.h"
#include "jstruct.h"
#include <cstring>
#include <mex.h>

/*
	prhs[0] - string					- string (char *)		- path to cover image
	prhs[1] - string					- string (char *)		- path to stego image
	prhs[2] - payload					- single (float)
	prhs[3] - struct config
				config.STC_h			- uint8 (unsigned int)	- default 0		- 0 for optimal simulator, otherwise STC submatrix height (try 7-12)
				config.seed				- int					- default 0		- random seed
*/
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
	const mxArray *m_cover_path;
	const mxArray *m_stego_path;

	// Default config
	float c_payload;
	int c_randSeed = 0;
	unsigned int c_stc_constr_height = 0;

	if ((nrhs != 3) && (nrhs != 4))
		mexErrMsgTxt ("Three or four inputs are required.\n3 inputs - [path to cover image (string)] [path to stego image (string)] [payload (single)] \n3 inputs - [path to cover image (string)] [path to stego image (string)] [payload (single)] [struct config]");

	if  (mxIsChar(prhs[0]))
		m_cover_path = prhs[0];
	else
		mexErrMsgTxt ("The first input (cover image path) must be a 'string'.");

	if  (mxIsChar(prhs[1]))
		m_stego_path = prhs[1];
	else
		mexErrMsgTxt ("The second input (stego image path) must be a 'string'.");

	if  ((mxIsClass(prhs[2], "single")) && (mxGetM(prhs[2])== 1) && (mxGetN(prhs[2])== 1))
		c_payload = (float)mxGetScalar(prhs[2]);
	else
		mexErrMsgTxt ("The third input (payload) must be a number of type 'single'.");

	if (nrhs == 4)
	{
		const mxArray *mat_config = prhs[3];

		int nfields = mxGetNumberOfFields(mat_config);
		if (nfields==0) mexErrMsgTxt ("The config structure is empty.");
		for(int fieldIndex=0; fieldIndex<nfields; fieldIndex++)
		{
			const char *fieldName = mxGetFieldNameByNumber(mat_config, fieldIndex);
			const mxArray *fieldContent = mxGetFieldByNumber(mat_config, 0, fieldIndex);

			if ((mxGetM(fieldContent)!= 1) || (mxGetN(fieldContent)!= 1))
				mexErrMsgTxt ("All config fields must be scalars.");
			// if every field is scalar
			if (strcmp(fieldName, "STC_h") == 0)
				if (mxIsClass(fieldContent, "uint32")) c_stc_constr_height = (unsigned int)mxGetScalar(fieldContent);
				else mexErrMsgTxt ("'config.STC_h' must be of type 'uint32'");
			if (strcmp(fieldName, "seed") == 0)
				if (mxIsClass(fieldContent, "int32")) c_randSeed = (int)mxGetScalar(fieldContent);
				else mexErrMsgTxt ("'config.seed' must be of type 'int32'");
		}
	}

	int buflen = (int)(mxGetM(m_cover_path) * mxGetN(m_cover_path)) + 1;
	char * buf = (char*)mxCalloc(buflen, sizeof(char));
	int status = mxGetString(m_cover_path, buf, buflen);
	if (status != 0)
   		mexErrMsgTxt("Could not convert input to a string.");
	std::string c_cover_path = buf;

	buflen = (int)(mxGetM(m_stego_path) * mxGetN(m_stego_path)) + 1;
	buf = (char*)mxCalloc(buflen, sizeof(char));
	status = mxGetString(m_stego_path, buf, buflen);
	if (status != 0)
   		mexErrMsgTxt("Could not convert input to a string.");
	std::string c_stego_path = buf;

	// Load JPEG image
	jstruct * coverStruct = new jstruct(c_cover_path, true);
	if (coverStruct->coef_arrays.size() != 1) { mexErrMsgTxt("Error: Only grayscale images can be embedded.");	}
	
	// Embedding
	cost_model_config *c_config = new cost_model_config(c_payload, false, 1, c_stc_constr_height, c_randSeed);
	base_cost_model * model = (base_cost_model *)new cost_model(coverStruct, c_config);

	float c_alpha_out, c_coding_loss_out = 0, c_distortion = 0;
	unsigned int c_stc_trials_used = 0;
	mat2D<int> * cover = coverStruct->coef_arrays[0];
	coverStruct->coef_arrays[0] = model->Embed(c_alpha_out, c_coding_loss_out, c_stc_trials_used, c_distortion);
	delete model;
	delete c_config;

	coverStruct->jpeg_write(c_stego_path, true);
	delete coverStruct;

	// Create distortion for matlab
	mwSize structSize[2];
	structSize[0] = 1;
	structSize[1] = 1;
	mxArray *mat_distortion = mxCreateNumericArray(2, structSize, mxSINGLE_CLASS, mxREAL);
	((float *)mxGetPr(mat_distortion))[0] = c_distortion;
	plhs[0] = mat_distortion;
} 