#ifndef LP_HH
#define LP_HH

#include <iostream>
#include <vector>
#include "lp.h"
#include "data.h"
#include "flt.h"

using namespace std;

inline NDArray lp::generate_one_hot(Shape shape, vector <string> label, vector <data::annotation> condition, Context ctx){
	
	int nbatch = shape[0];

	int nobject = shape[1];

	int nclass = shape[2];

	if (nbatch != condition.size()){
		
		cout << "batch size is not correct" << endl;

	}
	

	float * array = new float [shape.Size()] {0};

	if (condition.size() == nbatch){

		for (int b = 0; b != nbatch; ++b){ // iterate along batch
			
			//cout << "[";

			for (int o = 0; o != condition[b].vs.size(); ++o){ // iterate along number of alphabet in a plate (object)
				
				//cout << condition[b].vs[o];

				for (int l = 0; l != nclass; ++l){ // iterate along number of label
					
					
					if (condition[b].vs[o] == label[l]) //
							
						array[((b * nobject) + o) * nclass + l] = 1;
					
				}
			}
			//cout << "]" << endl;
		}
	}

	else if (condition.size() == 0)

		cout << "randomize object" << endl;

	else

		cout << "the size of given condition is not same as given number n" << endl;

	return flt::fmx::nd::FArray_to_NDArray(array, Shape(shape[0], shape[1] * shape[2]), ctx);
	
}


inline NDArray lp::generate_one_hot(float * farray, int batch, int n, vector <string> label, vector <data::annotation> condition, Context ctx){
	
	if (batch != condition.size()){
		
		cout << "batch size is not correct" << endl;

	}

	int size = label.size();

	cout << "label : " << label.size() << endl;

	cout << "n : " << n << endl;

	cout << "b : " << batch << endl;

	cout << "cond :" << condition.size() << endl;
	
	cout << condition[0].vs[5] << endl;

	if (condition.size() == batch){

		for (int b = 0; b != batch; ++b) // iterate along batch
		
			for (int i = 0; i != n; ++i){ // iterate along number of alphabet in a plate
				
				for (int l = 0; l != label.size(); ++l){ // iterate along number of label
					
					if (condition[b].vs[i] == label[l]) //
							
						farray[((b * n) + i) * label.size() + l] = 1;
					
				}
			}

	}

	else if (condition.size() == 0)

		cout << "randomize object" << endl;

	else

		cout << "the size of given condition is not same as given number n" << endl;

	return flt::fmx::nd::FArray_to_NDArray(farray, Shape(batch, n * label.size()), ctx);
	
}

/*
inline void lp::generate_one_hot(float * farray, int batch, int n, vector <string> label, vector <data::annotation> condition){
	
	if (batch != condition.size()){
		
		cout << "batch size is not correct" << endl;

	}

	int size = label.size();

	//farray = {1.2}; // label size * number of object
	
	cout << "label : " << label.size() << endl;

	cout << "n : " << n << endl;

	cout << "b : " << batch << endl;

	cout << "cond :" << condition.size() << endl;
	
	cout << condition[0].vs[5] << endl;

	if (condition.size() == batch){

		for (int b = 0; b != batch; ++b) // iterate along batch
		
			for (int i = 0; i != n; ++i){ // iterate along number of alphabet in a plate
				
				for (int l = 0; l != label.size(); ++l){ // iterate along number of label
					
					if (condition[b].vs[i] == label[l]) //
							
						farray[((b * n) + i) * label.size() + l] = 1;
					
				}
			}

	}

	else if (condition.size() == 0)

		cout << "randomize object" << endl;

	else

		cout << "the size of given condition is not same as given number n" << endl;
	
}

*/

vector <vector <char>> alphabet(int batch, int n){
	


}



#endif
