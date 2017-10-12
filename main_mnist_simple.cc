#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "flt.h"
#include "cgan_mnist_simple.hh"
#include "config.h"
#include <random>

using namespace std;
using namespace mxnet::cpp;

namespace bpo = boost::program_options;

int main(int argc, char ** argv){

	config c;

	c.size.height = 28;
	
	c.size.width = 28;
	
	c.nbatch = 128;
	
	c.nnoise = 100;

	
	c.pretrained = ("vgg16_weights.h5");
	
	c.debug = true;
	
	c.device = DeviceType::kGPU;
	

	flt::fdebug::log("before cgan instance ...", c.debug);
	
	CGAN_MNIST cgan(c);

	flt::fdebug::log("after cgan instance ...", c.debug);

	cgan.build();
	
	cgan.train(10000);

	return 0;
}


