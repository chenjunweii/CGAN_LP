#ifndef NETWORK_CC
#define NETWORK_CC

#include <iostream>
#include <vector>
#include <string>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include "flt.h"
#include "network.h"

using namespace std;
using namespace mxnet::cpp;
using namespace flt::fmx;

inline void network::VGG16_Deprecated(char * p, char * inputs, int nbatch, map <string, Symbol> *nodes, map <string, Symbol> *weight, map <string, Symbol> *bias, cv::Size size){
	
	layer::conv(p, "conv1_1", inputs, nodes, weight, bias, 64, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::conv(p, "conv1_2", "conv1_1", nodes, weight, bias, 64, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool1", "conv1_2", nodes);
	
	layer::conv(p, "conv2_1", "pool1", nodes, weight, bias, 128, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::conv(p, "conv2_2", "conv2_1", nodes, weight, bias, 128, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool2", "conv2_2", nodes);
	

	layer::conv(p, "conv3_1", "pool2", nodes, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv3_2", "conv3_1", nodes, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv3_3", "conv3_2", nodes, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool3", "conv3_3", nodes, Shape(2,2), Shape(2,2), Shape(1,1));
	
	layer::conv(p, "conv4_1", "pool3", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv4_2", "conv4_1", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv4_3", "conv4_2", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool4", "conv4_3", nodes, Shape(3,3), Shape(1,1), Shape(0,0));

	layer::conv(p, "conv5_1", "pool4", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv5_2", "conv4_1", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	layer::conv(p, "conv5_3", "conv4_2", nodes, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));

	layer::maxpool(p, "pool5", "conv5_3", nodes, Shape(3,3), Shape(2,2), Shape(0,0));
	
	//(*nodes)["pool5_reshape"] = Reshape((*nodes)["pool5"], Shape(nbatch, -1));
	cout << "hhas" << endl;	
	vector <Symbol> condition_vector_d = {Reshape((*nodes)[string(p) + "pool5"], Shape(nbatch, -1)), (*nodes)["fcondition"]};

	layer::concat(p, "condition_concat_d", &condition_vector_d, nodes, 1);
	cout << "aawd" << endl;
	layer::fullyconnected(p, "fc1", "condition_concat_d", nodes, weight, bias, 4096);
	
	(*nodes)[string(p) + "fc1_sigmoid"] = sigmoid((*nodes)[string(p) + "fc1"]);

	layer::fullyconnected(p, "fc2", "fc1", nodes, weight, bias, 4096);

	(*nodes)[string(p) + "fc2_sigmoid"] = sigmoid((*nodes)[string(p) + "fc2"]);
	
	layer::fullyconnected(p, "decision", "fc2_sigmoid", nodes, weight, bias, 1);

	(*nodes)[string(p) + "decision_sigmoid"] = sigmoid((*nodes)[string(p) + "decision"]);
}


inline Symbol network::VGG16(Symbol *inputs, Symbol *condition, int nbatch, map <string, Symbol> *weight, map <string, Symbol> *bias, cv::Size size){
	
	auto conv1_1 = layer::conv("conv1_1", (*inputs), weight, bias, 64, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv1_2 = layer::conv("conv1_2", conv1_1, weight, bias, 64, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool1 = layer::maxpool("pool1", conv1_2);
	
	auto conv2_1 = layer::conv("conv2_1", pool1, weight, bias, 128, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv2_2 = layer::conv("conv2_2", conv2_1, weight, bias, 128, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool2 = layer::maxpool("pool2", conv2_2);
	

	auto conv3_1 = layer::conv("conv3_1", pool2, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv3_2 = layer::conv("conv3_2", conv3_1, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv3_3 = layer::conv("conv3_3", conv3_2, weight, bias, 256, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool3 = layer::maxpool("pool3", conv3_3, Shape(2,2), Shape(2,2), Shape(1,1));
	
	auto conv4_1 = layer::conv("conv4_1", pool3, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv4_2 = layer::conv("conv4_2", conv4_1, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv4_3 = layer::conv("conv4_3", conv4_2, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool4 = layer::maxpool("pool4", conv4_3, Shape(3,3), Shape(1,1), Shape(0,0));

	auto conv5_1 = layer::conv("conv5_1", pool4, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv5_2 = layer::conv("conv5_2", conv5_1,  weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto conv5_3 = layer::conv("conv5_3", conv5_2, weight, bias, 512, Shape(3,3), Shape(1,1), Shape(1,1));
	auto pool5 = layer::maxpool("pool5", conv5_3, Shape(3,3), Shape(2,2), Shape(0,0));
	

	auto pool5_reshape = Reshape(pool5, Shape(nbatch, -1));
	
	vector <Symbol> features_vector {pool5_reshape, (*condition)};

	auto features = layer::concat("features", &features_vector,  1);

	auto fc1 = layer::fullyconnected("fcsggxxwwfs11", features, weight, bias, 4096);
	
	auto fc1_sigmoid = LeakyReLU(fc1) / 10;

	auto fc2 = layer::fullyconnected("fcss22", fc1_sigmoid, weight, bias, 4096);

	auto fc2_sigmoid = LeakyReLU(fc2) / 10;
	
	auto fc3 = layer::fullyconnected("fcss33", fc2_sigmoid, weight, bias, 1);

	auto fc3_sigmoid = sigmoid(fc3);

	return fc3_sigmoid;


}

inline void network::DEVGG16_Deprecated(char * p, char * inputs, map <string, Symbol> *nodes, map <string, Symbol> *weight, map <string, Symbol> *bias, cv::Size size){

	
	layer::deconv(p, "deconv5_3", inputs, nodes, weight, bias, 512);
	layer::deconv(p, "deconv5_2", "deconv5_3", nodes, weight, bias, 512);
	layer::deconv(p, "deconv5_1", "deconv5_2", nodes, weight, bias, 512, Shape(3,3), Shape(2,2));
	
	layer::maxpool(p, "depool5", "deconv5_1", nodes, Shape(2,2), Shape(1,1), Shape(1,1));
	cout << "hhh" << endl;
	layer::deconv(p, "deconv4_3", "depool5", nodes, weight, bias, 512);
	layer::deconv(p, "deconv4_2", "deconv4_3", nodes, weight, bias, 512);
	layer::deconv(p, "deconv4_1", "deconv4_2", nodes, weight, bias, 256, Shape(3,3), Shape(2,2));
	
	layer::maxpool(p, "depool4", "deconv4_1", nodes, Shape(2,2), Shape(1,1));

	layer::deconv(p, "deconv3_3", "depool4", nodes, weight, bias, 256);
	layer::deconv(p, "deconv3_2", "deconv3_3", nodes, weight, bias, 256);
	layer::deconv(p, "deconv3_1", "deconv3_2", nodes, weight, bias, 128, Shape(3,3), Shape(2,2));
	
	layer::maxpool(p, "depool3", "deconv3_1", nodes, Shape(2,2), Shape(1,1));

	layer::deconv(p, "deconv2_2", "depool3", nodes, weight, bias, 128);
	layer::deconv(p, "deconv2_1", "deconv2_2", nodes, weight, bias, 64, Shape(3,3), Shape(2,2));
	
	layer::maxpool(p, "depool2", "deconv2_1", nodes, Shape(2,2), Shape(1,1));
	
	layer::deconv(p, "deconv1_2", "depool2", nodes, weight, bias, 64, Shape(3,3), Shape(1,1));
	layer::deconv(p, "generated", "deconv1_2", nodes, weight, bias, 3, Shape(3,3), Shape(2,2));
	
	//layer::maxpool("depool1", (*nodes)[string("deconv1_1")], nodes, Shape(2,2), Shape(1,1), Shape(-1,-1));
	//layer::deconv("generated", (*nodes)["depool1"], nodes, weight, bias, 3);

}


inline Symbol network::DEVGG16(Symbol * inputs, map <string, Symbol> * weight, map <string, Symbol> * bias, cv::Size size){


	auto deconv5_3 = layer::deconv("deconv5_3", (*inputs), weight, bias, 512);
	auto deconv5_2 = layer::deconv("deconv5_2", deconv5_3, weight, bias, 512);
	auto deconv5_1 = layer::deconv("deconv5_1", deconv5_2, weight, bias, 512, Shape(3,3), Shape(2,2));
	
	//auto depool5 = layer::maxpool("depool5", deconv5_1, Shape(2,2), Shape(1,1), Shape(1,1));
	
	auto deconv4_3 = layer::deconv("deconv4_3", deconv5_1, weight, bias, 512);
	auto deconv4_2 = layer::deconv("deconv4_2", deconv4_3, weight, bias, 512);
	auto deconv4_1 = layer::deconv("deconv4_1", deconv4_2, weight, bias, 256, Shape(3,3), Shape(2,2));
	
	//auto depool4 = layer::maxpool("depool4", deconv4_1, Shape(2,2), Shape(1,1));

	auto deconv3_3 = layer::deconv("deconv3_3", deconv4_1, weight, bias, 256);
	auto deconv3_2 = layer::deconv("deconv3_2", deconv3_3, weight, bias, 256);
	auto deconv3_1 = layer::deconv("deconv3_1", deconv3_2, weight, bias, 128, Shape(3,3), Shape(2,2));
	
	//auto depool3 = layer::maxpool("depool3", deconv3_1, Shape(2,2), Shape(1,1));

	auto deconv2_2 = layer::deconv("deconv2_2", deconv3_1, weight, bias, 128);
	
	auto deconv2_1 = layer::deconv("deconv2_1", deconv2_2, weight, bias, 64, Shape(3,3), Shape(2,2));
	
	//auto depool2 = layer::maxpool("depool2", deconv2_1, Shape(2,2), Shape(1,1));
	
	auto deconv1_2 = layer::deconv("deconv1_2", deconv2_1, weight, bias, 64, Shape(3,3), Shape(1,1));
	
	auto deconv1_1 = layer::deconv("deconv1_1", deconv1_2, weight, bias, 3, Shape(3,3), Shape(2,2));
	
	//return clip(deconv1_1, 0, 1);
	return sigmoid(deconv1_1);
}

inline Symbol network::MLP(Symbol *inputs, Symbol *condition, int nbatch, map <string, Symbol> *weight, map <string, Symbol> *bias, cv::Size size){
	
	vector <Symbol> features_vector {(*inputs), (*condition)};

	auto features = layer::concat("features", &features_vector,  1);
	
	auto fc1 = layer::fullyconnected("fc1", features, weight, bias, 128);
	
	auto relu1 = relu(fc1);

	auto fc2 = layer::fullyconnected("fc2", relu1, weight, bias, 1);
	
	return sigmoid(fc2); // non linear activation function


}

inline Symbol network::DEMLP(Symbol * inputs, map <string, Symbol> * weight, map <string, Symbol> * bias, cv::Size size){

	auto fc1 = layer::fullyconnected("defc1", (*inputs), weight, bias, 128);
	
	auto relu1 = relu(fc1);

	auto fc2 = layer::fullyconnected("defc2", relu1, weight, bias, 784);
	
	return sigmoid(fc2); // linear (regression)
	
}


#endif
