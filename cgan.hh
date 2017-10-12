#ifndef CGAN_HH
#define CGAN_HH

#include <iostream>
#include <map>
#include <cmath>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
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

	checkpoint = c.checkpoint;

};


inline void CGAN_LP::build(){			
	
	cout << "Start Build" << endl;

	auto lp_reshape = Reshape(lp, Shape(nbatch, -1));

	node["lp_reshape"] = lp_reshape;
	
	condition[0] = lp_reshape;

	condition[1] = z;

	/* 
	 *	condition => [nbatch, condition]
	 *	
	 *	condition_fc => map 2d "condition" [nbatch, condition] => [nbatch, channel = 128 , height, width] => deconvolution
	 */

	auto condition_concat = layer::concat("condition_concat", &condition, 1);

	auto fc1 = LeakyReLU(layer::fullyconnected("defc1", condition_concat, &weight, &bias, 256 * 2 * 2));
	
	auto fc2 = LeakyReLU(layer::fullyconnected("defc2", fc1, &weight, &bias, 512 * 4 * 4));

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
	auto reconstruction_loss = mean((node["generated"] - inputs) * (node["generated"] - inputs));

	return MakeLoss("G_Loss", generation_loss);
}


inline Symbol CGAN_LP::D_Loss(){	

	Symbol real_decision_loss = mean(loss::cross_entropy(node["decision_real"],
				ones_like("real_loss", node["decision_real"])));
	
	Symbol fake_decision_loss = mean(loss::cross_entropy(node["decision_fake"],
				zeros_like("fake_loss", node["decision_fake"])));

	return MakeLoss("D_Loss", real_decision_loss + fake_decision_loss);
}


inline void CGAN_LP::train(int epoch){
	
	Context ctx(device, 0);
	
	data::db dataset(CGAN_LP::sdataset, CGAN_LP::slist, CGAN_LP::label, CGAN_LP::size, CGAN_LP::nbatch, true, data::MODE::generation);

	/* setup shape */

	map <string, map <string, vmx_shape>> inf; // input shape
	
	map <string, map <string, mx_shape>> arg;

	arg["e"]["inputs"] = {nbatch, size.height, size.width, 3};

	arg["e"]["lp"] = {nbatch, nobject * nclass};
	
	arg["e"]["z"] = {nbatch, nnoise};

	arg["d"] = arg["e"];

	arg["g"]["lp"] = {nbatch, nobject * nclass};
	
	arg["g"]["z"] = {nbatch, nnoise};
	
	
	map <string, NDArray> nd, grad;


	Uniform uniform(0, 1);

	Symbol g = G_Loss();//node["decision_real"];

	Symbol d = D_Loss();//D_Loss();
	
	Symbol e = g + d;

	//node["generated"].InferShape(arg["g"], &gen_in, &gen_aux, &gen_out);

	//cout << "Generated Image : " << Shape(gen_out[0]) << endl;
	
	e.InferShape(arg["e"], &inf["e"]["in"], &inf["e"]["aux"], &inf["e"]["out"]);
	
	d.InferShape(arg["d"], &inf["d"]["in"], &inf["d"]["aux"], &inf["d"]["out"]);
	
	g.InferShape(arg["g"], &inf["g"]["in"], &inf["g"]["aux"], &inf["g"]["out"]);
	
	//node["generated"].InferShape(gs, &gen_in, &gen_aux, &gen_out);

	//cout << "Generated Image : " << Shape(gen_out[0]) << endl;
	
	map <string, vector <string>> nnode;

	nnode["e"] = e.ListArguments();
	
	nnode["g"] = g.ListArguments();

	nnode["d"] = d.ListArguments();
	
	
	
	/* setup weight */

	/* Deprecated */
	
	cout << "init " << endl;
	
	string weight_file;
	
	init::init_mode modes;

	int iters_chkp;

	if (checkpoint != ""){
		
		modes = init::init_mode::restore;

		weight_file = checkpoint;

		vector <string> strs;
		
		boost::split(strs, checkpoint, boost::is_any_of("/."));

		iters_chkp = stoi(strs[1]);

	}

	else{ 

		modes = init::init_mode::pretrained;

		weight_file = pretrained;

		iters_chkp = 0;

	}

	init::init_weight_simple(nnode["e"], inf["e"]["in"], nd, grad, ctx, modes, weight_file);

	dataset.next();
	
	nd["z"] = NDArray(Shape(nbatch, nnoise), ctx);
	
	uniform("z", &nd["z"]);
	
	nd["inputs"] = fimage::MatVector_to_NDArray(dataset.inputs, ctx);		
	
	nd["lp"] = lp::generate_one_hot(Shape(nbatch, nobject, nclass), label, dataset.target, ctx);

	//auto wrt_g = init::wrt(vector <char * > {"de"}, g_node_list);
	//auto wrt_d = init::wrt(vector <char *> {"conv", "fc"}, d_node_list);

	//auto wrt_end = init::wrt(vector <char *> {"de", "conv", "fc"}, end_node_list);
	
	/* setup executor */
	
	Executor * S = fimage::decodeb(node["generated"]).SimpleBind(ctx, nd);
	
	Executor * G = g.SimpleBind(ctx, nd, grad);
	
	Executor * D = d.SimpleBind(ctx, nd, grad);
	
 	Optimizer * Gadam = OptimizerRegistry::Find("adam");
 	
	Optimizer * Dadam = OptimizerRegistry::Find("sgd");

	Gadam->SetParam("lr", 0.000001)
		->SetParam("clip_gradient", 0.05);
	
	Dadam->SetParam("lr", 0.00001)
		->SetParam("clip_gradient", 0.05);
	
	bool dswitch = true;

	bool gswitch = true;

	vector <float> d_loss(1, 10);

	vector <float> g_loss(1, 10);

	float d_loss_obj = 0.1;

	float g_loss_obj = 0.1;


	for (int i = iters_chkp; i != epoch; ++i){
				

		
		(*D).Forward(true);
		
		if (d_loss[0] > d_loss_obj or i != iters_chkp){
			
			(*D).Backward();
					
			for (int j = 0; j != (*D).arg_arrays.size(); ++j){
					
				string prefix = nnode["d"][j].substr(0,4);

				if ((prefix == "wcon") or (prefix == "bcon") or (prefix == "w_fc") or (prefix == "b_fc")){

//					cout << "D -> Update : " <<  nnode["d"][j] << (*D).grad_arrays[j].Reshape(Shape(-1)).Slice(0,1) << endl;
					
					(*Dadam).Update(j, (*D).arg_arrays[j], (*D).grad_arrays[j]);
				}
			}
		}

		
		(*G).Forward(true);
		
		if (g_loss[0] > g_loss_obj or i != iters_chkp){
			
			(*G).Backward();	
					
			for (int j = 0; j != (*G).arg_arrays.size(); ++j){
					
				string prefix = nnode["g"][j].substr(0,4);
				
				//cout << nnode["g"][j] << endl;
				
				if ((prefix == "wdec") or (prefix == "bdec") or (prefix == "w_de") or (prefix == "b_de")){
					
//					cout << "G -> Update : " <<  nnode["g"][j] << (*G).grad_arrays[j].Reshape(Shape(-1)).Slice(0,1) << endl;

					(*Gadam).Update(j, (*G).arg_arrays[j], (*G).grad_arrays[j]);


				}	
			}
		}


		vector <NDArray> dout = (*D).outputs;
	
		vector <NDArray> gout = (*G).outputs;
		
		dout[0].SyncCopyToCPU(d_loss.data(), 1);
		
		gout[0].SyncCopyToCPU(g_loss.data(), 1);
		
		if (i % 1 == 0)

			cout << "Epoch : " << i << ", D Loss : " << d_loss[0] << ", G Loss : " << g_loss[0] << endl;
		
		if (d_loss[0] <= d_loss_obj and g_loss[0] <= g_loss_obj){

			d_loss_obj /= 2.0;

			g_loss_obj /= 2.0;
		}

		else if (isnan(d_loss[0]) or isnan(g_loss[0]))

			break;
		
		if (i % 1000 == 0){
			
			uniform("z", &nd["z"]);
			
			(*S).Forward(false);
		
			vector <NDArray> sout = (*S).outputs;

			NDArray fig = sout[0].Slice(0,1);
			
			try{

				fimage::save("out/" + to_string(i) + ".jpg", fig, 255);

			}

			catch (...){

				cout << "Error occur when writing Image ... " << endl;
			}


		}
		//cout << d_loss[0] << endl;
		//cout << "G Loss : " << gout[0] << endl;
		
		//cout << "S out : " << Shape(sout[0].GetShape()) << endl;
		
		
		//fimage::saveb("out/" + to_string(i), i, nd_d[input_d]);
	
		dataset.next();
		
		uniform("z", &nd["z"]);
		
		nd["inputs"] = fimage::MatVector_to_NDArray(dataset.inputs, ctx);		
		
		nd["lp"] = lp::generate_one_hot(Shape(nbatch, nobject, nclass), label, dataset.target, ctx);		

		if (i % 5000 == 0){
			
			fhdf5 saver("model/" + to_string(i) + ".chk");

			saver.open();
			
			saver.save_NDArray(nd);

			//saver.save_NDArray(grad);
			
			saver.close();

		}
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
