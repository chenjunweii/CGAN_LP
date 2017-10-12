#ifndef LP_H
#define LP_H

#include <iostream>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>

namespace lp{

	//inline void generate_one_hot(float * farray, int batch, int n, vector <string> label, vector <data::annotation> condition);
	inline NDArray generate_one_hot(float * farray, int batch, int n, vector <string> label, vector <data::annotation> condition, Context ctx);
	
	inline NDArray generate_one_hot(Shape shape, vector <string> label, vector <data::annotation> condition, Context ctx);

}


#endif
