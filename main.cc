#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "flt.h"
#include "cgan.hh"
#include "config.h"
#include <random>

using namespace std;
using namespace mxnet::cpp;

namespace po = boost::program_options;

int main(int argc, char ** argv){
	
	po::options_description desc("Options");
	
	//desc.add_options() ("help", "Print help messages");
	
	desc.add_options() ("restore,r", po::value <string> () -> default_value(""), "restore checkpoint");
	
	po::variables_map vm;

	try {
        
		po::store(po::parse_command_line(argc, argv, desc), vm);
        
		po::notify(vm);

    } catch (po::error& e) {
        
		cerr << "ERROR: " << e.what() << endl << endl << desc << endl;
        
		return 1;
    }


	config c;

	c.size.height = 97;
	
	c.size.width = 97;
	
	c.nbatch = 10;
	
	c.nobject = 10;
	
	c.nnoise = 400;

	c.sdataset = string("Alphabet");
	
	c.pretrained = ("vgg16_weights.h5");

	c.checkpoint = vm["restore"].as <string> ();

	c.slist = string("train_multi.txt");

	c.debug = true;

	c.label = vector <string> {

		"Background",

		"A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
	
		"K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
		
		"V", "W", "X", "Y", "Z",

		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"

	};

	c.device = DeviceType::kGPU;

	flt::fdebug::log("before cgan instance ...", c.debug);
	
	CGAN_LP cgan(c);

	flt::fdebug::log("after cgan instance ...", c.debug);

	cgan.build();
	
	cgan.train(1000000);

	return 0;
}


