#ifndef CGAN_SIMPLE_HH
#define CGAN_SIMPLE_HH

#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "network.hh"
#include "config.h"
#include "cgan.h"
#include "data.hh"
#include "lp.hh"
#include "loss.hh"
#include "init.hh"

using namespace std;
using namespace mxnet::cpp;
using namespace flt::fmx;
namespace bpo = boost::program_options;



CGAN_LP::CGAN_LP(config c){

	sdataset = c.sdataset;

	slist = c.slist;

	label = c.label;

	size = c.size;

	nbatch = c.nbatch;

	device = c.device;

	nobject = c.nobject;
	
	nnoise = c.nnoise;

	nclass = label.size();

	pretrained = c.pretrained;

};


inline void CGAN_LP::build(){			
	
	cout << "Start Build" << endl;

	auto lp_reshape = Reshape(lp, Shape(nbatch, -1));
	
	condition[0] = lp_reshape;

	condition[1] = z;

	/* 
	 *	condition => [nbatch, condition]
	 *	
	 *	condition_fc => map 2d "condition" [nbatch, condition] => [nbatch, channel = 128 , height, width] => deconvolution
	 */

	auto condition_concat = layer::concat("condition_concat", &condition, 1);

	auto fc1 = sigmoid(layer::fullyconnected("defc1", condition_concat, &weight, &bias, 256 * 2 * 2));
	
	auto fc2 = sigmoid(layer::fullyconnected("defc2", fc1, &weight, &bias, 512 * 4 * 4));

	auto fc2_4d = Reshape(fc2, Shape(nbatch, 512, 4, 4)); 

	node["generated"] = network::DEVGG16(&fc2_4d, &weight, &bias, size);
	
	/*
	 *	Convolution VGG
	 *	
	 *	Generated Image => flattened feature map + flattened condition => fullyconnected layer => real or fake
	 *
	 *
	 */

	node["decision_real"] = network::VGG16(&inputs, &lp, nbatch, &weight, &bias, size); // presigmoid
	
	node["decision_fake"] = network::VGG16(&node["generated"], &lp, nbatch, &weight, &bias, size); // presigmoid
	
};

inline Symbol CGAN_LP::Loss(){

	Symbol generate_loss = mean(loss::cross_entropy(node["decision_fake"],
				ones_like("generate_loss", node["decision_fake"])));
	
	Symbol real_decision_loss = mean(loss::cross_entropy(node["decision_real"],
				ones_like("real_loss", node["decision_real"])));
	
	Symbol fake_decision_loss = mean(loss::cross_entropy(node["decision_fake"],
				zeros_like("fake_loss", node["decision_fake"])));
		
	return MakeLoss("endloss", generate_loss + fake_decision_loss + real_decision_loss);
};

inline Symbol CGAN_LP::G_Loss(){

	/* let generated image can be considered a real image, so use ones_like, not zeros_like */

	auto generation_loss = mean(loss::cross_entropy(node["decision_fake"], ones_like("generate_loss", node["decision_fake"])));
	//
	auto reconstruction_loss = mean(loss::cross_entropy(node["generated"], inputs));

	return MakeLoss("G_Loss", generation_loss);
}


inline Symbol CGAN_LP::D_Loss(){	

	Symbol real_decision_loss = mean(loss::cross_entropy(node["decision_real"],
				ones_like("real_loss", node["decision_real"])));
	
	Symbol fake_decision_loss = mean(loss::cross_entropy(node["decision_fake"],
				zeros_like("fake_loss", node["decision_fake"])));

	return MakeLoss("D_Loss", real_decision_loss * 0.5 + 0.5 * fake_decision_loss);
}


inline void CGAN_LP::train(int epoch){
	
	Context ctx(device, 0);
	
	data::db dataset(CGAN_LP::sdataset, CGAN_LP::slist, CGAN_LP::label, CGAN_LP::size, CGAN_LP::nbatch, true, data::MODE::generation);

	dataset.next();
	
	//cv::Mat a = dataset.inputs[0];

	//cv::imshow("windows", a);

	//cv::waitKey(0);
	
	/* setup shape */

	vmx_shape endin, gin, din, gen_in; // input shape
	
	vmx_shape endaux, gaux, daux, gen_aux;
	
	vmx_shape endout, gout, dout, gen_out;
	
	map <string, mx_shape> ends;
	map <string, mx_shape> ds;
	map <string, mx_shape> gs;


	
	ends["inputs"] = {nbatch, size.height, size.width, 3};

	ends["lp"] = {nbatch, nobject * nclass};
	
	ends["z"] = {nbatch, nnoise};

	ds = ends;


	gs["lp"] = {nbatch, nobject * nclass};
	
	gs["z"] = {nbatch, nnoise};
	
	//gs["inputs"] = {nbatch, size.height, size.width, 3};

	/* setup grad map*/
	
	
	//map <string, NDArray> grad;

	//map <string, OpReqType> op;
	

	/* setup ndarray */

	//ndarg["inputs"] = fimage::MatVector_to_NDArray(dataset.inputs, ctx);

	//cout << "Inputs : " << Shape(ndarg["inputs"].GetShape()) << endl;
	
	//ndarg["z"] = NDArray(Shape(nbatch, nnoise), ctx);

	//ndarg["lp"] = lp::generate_one_hot(Shape(nbatch, nobject, nclass), label, dataset.target, ctx);
	


	Uniform uniform(0, 1);

	//uniform("z", &ndarg["z"]);
	

	/*
	 * Decision Real => no problem
	 *
	 * Decision Fake => error
	 *
	 * Geneartaed Image => no problem
	 * 
	 *
	 * */
	
	cout << "loss" << endl;

	Symbol end = Loss();
	
	Symbol g = G_Loss();//node["decision_real"];

	Symbol d = D_Loss();
	
	/*vector <string> gss = g.ListArguments();
	
	for(auto &i : gss)

		cout << i << endl;
	*/

	node["generated"].InferShape(gs, &gen_in, &gen_aux, &gen_out);

	cout << "Generated Image : " << Shape(gen_out[0]) << endl;
	
	end.InferShape(ends, &endin, &endaux, &endout);
	
	g.InferShape(gs, &gin, &gaux, &gout);
	
	d.InferShape(ds, &din, &daux, &dout);
	
	//node["generated"].InferShape(gs, &gen_in, &gen_aux, &gen_out);

	//cout << "Generated Image : " << Shape(gen_out[0]) << endl;

	vector <string> end_node_list = end.ListArguments();
	
	vector <string> g_node_list = g.ListArguments();

	vector <string> d_node_list = d.ListArguments();
	
	
	//vector <NDArray> aux_d(d_node_list.size());

	//vector <NDArray> aux_g(g_node_list.size());
	
	
	//map <string, Shape> node_shape = merge_uint_vector(node_name, in_shape);
	
	nd_end = vector <NDArray> (end_node_list.size());

	grad_end = vector <NDArray> (end_node_list.size());
	
	nd_g = vector <NDArray> (g_node_list.size());

	grad_g = vector <NDArray> (g_node_list.size());

	nd_d = vector <NDArray> (d_node_list.size());

	grad_d = vector <NDArray> (d_node_list.size());



	//auto iinput = find(end_node_list.begin(), end_node_list.end(), "inputs") - end_node_list.begin();
	
	//nd_end[iinput] = fimage::MatVector_to_NDArray(dataset.inputs, ctx);

	//auto iz = find(end_node_list.begin(), end_node_list.end(), "z") - end_node_list.begin();

	auto endz = find(end_node_list.begin(), end_node_list.end(), "z") - end_node_list.begin();
	
	auto gz = find(g_node_list.begin(), g_node_list.end(), "z") - g_node_list.begin();
	
	auto dz = find(d_node_list.begin(), d_node_list.end(), "z") - d_node_list.begin();
	
	//nd_end[iz] = NDArray(Shape(nbatch, nnoise), ctx);
	
	nd_end[endz] = NDArray(Shape(nbatch, nnoise), ctx);
	
	uniform("z", &nd_end[endz]);
	
	//cout << "ss" << endl;
	//nd_d[dz] = nd_end[endz];

	//nd_g[gz] = nd_end[endz];




	
	auto endlp = find(end_node_list.begin(), end_node_list.end(), "lp") - end_node_list.begin();
	
	
	auto glp = find(g_node_list.begin(), g_node_list.end(), "lp") - g_node_list.begin();
	
	auto dlp = find(d_node_list.begin(), d_node_list.end(), "lp") - d_node_list.begin();
	
	nd_end[endlp] = lp::generate_one_hot(Shape(nbatch, nobject, nclass), label, dataset.target, ctx);
	
	//nd_d[dlp] = nd_end[endlp];

	//nd_g[glp] = nd_end[endlp];
	
	
	auto input_end = find(end_node_list.begin(), end_node_list.end(), "inputs") - end_node_list.begin();

	auto input_d = find(d_node_list.begin(), d_node_list.end(), "inputs") - d_node_list.begin();

	//auto input_g = find(g_node_list.begin(), g_node_list.end(), "inputs") - g_node_list.begin();

	nd_end[input_end] = fimage::MatVector_to_NDArray(dataset.inputs, ctx);
	
	//nd_d[input_d] = nd_end[input_end];


	/* setup weight */

	/* Deprecated */
	
	

	init::init_weight(end_node_list, endin, nd_end, grad_end, ctx, init::init_mode::pretrained, pretrained);

	for(int i = 0; i != d_node_list.size(); ++i){
		
		auto idx = find(end_node_list.begin(), end_node_list.end(), d_node_list[i]) - end_node_list.begin();
		
		nd_d[i] = nd_end[idx];
		
		grad_d[i] = grad_end[idx];

	}


	for(int i = 0; i != g_node_list.size(); ++i){

		auto idx = find(end_node_list.begin(), end_node_list.end(), g_node_list[i]) - end_node_list.begin();
		
		nd_g[i] = nd_end[idx];
		
		grad_g[i] = grad_end[idx];
	}
	
	
	
	auto wrt_g = init::wrt(vector <char * > {"de"}, g_node_list);

	auto wrt_d = init::wrt(vector <char *> {"conv", "fc"}, d_node_list);

	auto wrt_end = init::wrt(vector <char *> {"de", "conv", "fc"}, end_node_list);
	
	/*for(auto &i : node_shape){
		
		cout << "[" << i.first << "] : " << i.second << ", ND : " << Shape(ndarg[i.first].GetShape())<< endl;
	}*/
	
	/* setup executor */

	//Executor * G = g.SimpleBind(ctx, ndarg, grad);
	
	//Executor * D = d.SimpleBind(ctx, ndarg, grad);
	
	cout << "Create Executor ... " << endl;

	Executor * S = fimage::decodeb(node["generated"]).Bind(ctx, nd_g, grad_g, wrt_g, vector <NDArray> ());
	
	Executor * D = d.Bind(ctx, nd_d, grad_d, wrt_d, vector <NDArray> ());
	
	Executor * G = g.Bind(ctx, nd_g, grad_g, wrt_g, vector <NDArray> ());
	
	
	cout << "Share Executor " << endl;

 	Optimizer * Gadam = OptimizerRegistry::Find("adam");
 	
	Optimizer * Dadam = OptimizerRegistry::Find("sgd");

	(*Gadam).SetParam("lr", 0.000001);
	
	(*Dadam).SetParam("lr", 0.00001);
	
	for (int i = 0; i != epoch; ++i){
			
		
	//	cout << nd_end[endlp].Slice(0, 1) << endl;
		//cout << nd_end[endz].Slice(0,1) << endl;
		//cout << nd_end[input_end].Reshape(Shape(-1,3)).Slice(0,1) << endl;
		
		//cout << nd_d[input_d].Reshape(Shape(-1,3)).Slice(0,1) << endl;

		if (i % 100 == 0){

			(*S).Forward(false);
		
			vector <NDArray> sout = (*S).outputs;
			
			fimage::saveb("out/" + to_string(i), i, sout[0]);

		}

		(*D).Forward(true);
		
		(*D).Backward();
				
		for (int j = 0; j != grad_d.size(); ++j){
				
			string prefix = d_node_list[j].substr(0,2);

			if ((prefix == "wc") or (prefix == "bc") or (prefix == "wf") or (prefix == "bf")){

				//auto match = std::find(end_node_list.begin(), end_node_list.end(), d_node_list[i]) - end_node_list.begin();
				//cout << "[D] : " << d_node_list[i] << endl;

				//(*Dadam).Update(match, (*D).arg_arrays[i], (*D).grad_arrays[i]);

				(*Dadam).Update(j, (*D).arg_arrays[j], (*D).grad_arrays[j]);
			}
		}	
		
		
		(*G).Forward(true);

		(*G).Backward();	
				
		for (int j = 0; j != grad_g.size(); ++j){
				
			string prefix = g_node_list[j].substr(0,2);
				
			if ((prefix == "wd") or (prefix == "bd")){
				
				//cout << "[G] : " << g_node_list[i] << endl;
				//auto match = std::find(end_node_list.begin(), end_node_list.end(), g_node_list[i]) - end_node_list.begin();
				//(*Gadam).Update(match, (*G).arg_arrays[i], (*G).grad_arrays[i]);
				(*Gadam).Update(j, (*G).arg_arrays[j], (*G).grad_arrays[j]);
			}
			
		}
		
		

		vector <NDArray> dout = (*D).outputs;
	
		vector <NDArray> gout = (*G).outputs;
		
		//cout << "G Loss : " << gout[0] << endl;
		
		//cout << "S out : " << Shape(sout[0].GetShape()) << endl;

		cout << "Epoch : " << i << ", D Loss : " << dout[0] << ", G Loss : " << gout[0] << endl;
		
		//fimage::saveb("out/" + to_string(i), i, nd_d[input_d]);
	
		dataset.next();
		
		nd_end[input_end] = fimage::MatVector_to_NDArray(dataset.inputs, ctx);		
		
		nd_d[input_d] = nd_end[input_end];
		
		/* for reconstruction , not generation */

		//auto uginput = find(g_node_list.begin(), g_node_list.end(), "inputs") - g_node_list.begin();
		
	//	nd_g[input_g] = nd_end[input_end];

	
		uniform("z", &nd_end[endz]);
		nd_g[gz] = nd_end[endz];		
		nd_d[dz] = nd_end[endz];

		
		nd_end[endlp] = lp::generate_one_hot(Shape(nbatch, nobject, nclass), label, dataset.target, ctx);
		nd_g[glp] = nd_end[endlp];
		nd_d[dlp] = nd_end[endlp];
		
	}
	
	delete G;

	delete D;

	delete S;

	MXNotifyShutdown();

};

CGAN_LP::~CGAN_LP(){
	
//	MXNotifyShutdown();

}

/*
	ndarg["lp"] = lp::generate_one_hot(Shape(nbatch, nobject, nclass), label, dataset.target, ctx);
	
	cout << "LP Shape : " << Shape(ndarg["lp"].GetShape()) << endl;
	
	float * testa = new float [Shape(ndarg["lp"].GetShape()).Size()];
	
	flt::fmx::nd::NDArray_to_FArray(&ndarg["lp"], testa, Shape(ndarg["lp"].GetShape()));
	
	for (int i = 0; i != nbatch; ++i){
		
		cout << "Batch[" << i << "] : [";

		for (int j = 0; j != nobject; j++){

			cout << "A : ";
			
			for (int k = 0; k != nclass; k++){
				
				cout << testa[((i * nobject) + j) * nclass + k] << ", ";
			}

			cout << endl;
		}
		cout << "]" << endl;
	}
*/
#endif
