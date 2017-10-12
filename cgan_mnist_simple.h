#ifndef CGAN_MNIST_SIMPLE_H
#define CGAN_MNIST_SIMPLE_H

#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "network.hh"

#include "config.h"

using namespace std;
using namespace mxnet::cpp;

namespace bpo = boost::program_options;

typedef vector <mx_uint> mx_shape;

typedef vector <vector <mx_uint>> vmx_shape; // input shape



class CGAN_MNIST{
	
	public:

		map <string, Symbol> node;
		
		map <string, Symbol> weight;
		
		map <string, Symbol> bias;
		
		map <string, NDArray> ndarg;
		
		vector <NDArray> nd_end;

		vector <NDArray> grad_end;

		vector <NDArray> nd_g;

		vector <NDArray> grad_g;

		vector <NDArray> nd_d;

		vector <NDArray> grad_d;

		map <string, NDArray> fixed;

		map <string, vector <mx_uint>> inputs_shape;
		
		Symbol mode;
		
		cv::Size size;

		string slist;

		string sdataset;
		
		string pretrained;

		vector <string> label;

		int nbatch; // number of training batch
		
		int nobject; // number of object
		
		int nclass; // number of classes
		
		int nnoise; // noise size
		
		DeviceType device;
		
		Symbol inputs = Symbol::Variable("inputs");
		
		Symbol z = Symbol::Variable("z");

		Symbol lp = one_hot(Symbol::Variable("lp"), 10);
		
//		Symbol generated = Symbol::Variable("generated"); // not same as node["generated"]

		vector <Symbol> condition = vector <Symbol> (2);// = Symbol::Variable("condition");
	
		CGAN_MNIST (config c);

		~CGAN_MNIST();

		inline void build();

		inline void train(int iters);

		inline Symbol Loss();

		inline Symbol D_Loss();
		
		inline Symbol G_Loss();
};

#endif
