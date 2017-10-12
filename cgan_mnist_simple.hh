#ifndef CGAN_MNIST_SIMPLE_HH
#define CGAN_MNIST_SIMPLE_HH

#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "network.hh"
#include "config.h"
#include "cgan_mnist_simple.h"
#include "data.hh"
#include "lp.hh"
#include "loss.hh"
#include "init.hh"

using namespace std;
using namespace mxnet::cpp;
using namespace flt::fmx;
namespace bpo = boost::program_options;



CGAN_MNIST::CGAN_MNIST(config c){

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


inline void CGAN_MNIST::build(){			
	
	condition[0] = lp;

	condition[1] = z;

	auto condition_concat = layer::concat("condition_concat", &condition, 1);

	node["generated"] = network::DEMLP(&condition_concat, &weight, &bias, size);
	
	node["decision_real"] = network::MLP(&inputs, &lp, nbatch, &weight, &bias, size);

	node["decision_fake"] = network::MLP(&node["generated"], &lp, nbatch, &weight, &bias, size);

};

inline Symbol CGAN_MNIST::Loss(){

	Symbol generate_loss = mean(loss::cross_entropy(node["decision_fake"], ones_like("generate_loss", node["decision_fake"])));
	
	Symbol real_decision_loss = mean(loss::cross_entropy(node["decision_real"],	ones_like("real_loss", node["decision_real"])));
	
	Symbol fake_decision_loss = mean(loss::cross_entropy(node["decision_fake"], zeros_like("fake_loss", node["decision_fake"])));
		
	return MakeLoss("endloss", generate_loss + fake_decision_loss + real_decision_loss);
};

inline Symbol CGAN_MNIST::G_Loss(){

	/* let generated image can be considered a real image, so use ones_like, not zeros_like */

	auto generation_loss = mean(loss::cross_entropy(node["decision_fake"], ones_like("generate_ones_like_loss", node["decision_fake"])));
	
	//return MakeLoss("G_Loss", generation_loss);
	return MakeLoss("G_Loss", mean(0 - log(node["decision_fake"])));
}


inline Symbol CGAN_MNIST::D_Loss(){	

	Symbol real_decision_loss = mean(loss::cross_entropy(node["decision_real"], ones_like("real_one_loss", node["decision_real"])));
	
	Symbol fake_decision_loss = mean(loss::cross_entropy(node["decision_fake"],	zeros_like("fake_zero_loss", node["decision_fake"])));


	return MakeLoss("D_Loss", mean(0 - log(node["decision_real"]) - log(1 - node["decision_fake"])));
	//return MakeLoss("D_Loss", real_decision_loss + fake_decision_loss);
}


inline void CGAN_MNIST::train(int epoch){
	
	Context ctx(device, 0);
	
	
 	auto train_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./mnist_data/train-images-idx3-ubyte")
      .SetParam("label", "./mnist_data/train-labels-idx1-ubyte")
      .SetParam("batch_size", nbatch)
      .SetParam("flat", 1)
      .CreateDataIter();
 	auto val_iter = MXDataIter("MNISTIter")
      .SetParam("image", "./mnist_data/t10k-images-idx3-ubyte")
      .SetParam("label", "./mnist_data/t10k-labels-idx1-ubyte")
      .SetParam("batch_size", nbatch)
      .SetParam("flat", 1)
      .CreateDataIter();



	/* setup shape */

	vmx_shape endin, gin, din, gen_in; // input shape
	
	vmx_shape endaux, gaux, daux, gen_aux;
	
	vmx_shape endout, gout, dout, gen_out;
	
	map <string, mx_shape> ends, ds, gs;
	
	map <string, NDArray> map_end, map_d, map_g;

	map <string, NDArray> map_grad_end, map_grad_d, map_grad_g;
	
	ends["inputs"] = {nbatch, size.height * size.width};

	ends["lp"] = {nbatch};
	
	ends["z"] = {nbatch, nnoise};

	ds["inputs"] = {nbatch, size.height * size.width};

	ds["lp"] = {nbatch};

	ds["z"] = {nbatch, nnoise};


	gs["lp"] = {nbatch};
	
	gs["z"] = {nbatch, nnoise};
	
	
	Symbol g = G_Loss();//node["decision_real"];

	Symbol d = D_Loss();
	
	Symbol end = g + d;

	Uniform uniform(-1, 1);
	
	//node["generated"].InferShape(gs, &gen_in, &gen_aux, &gen_out);

	//g.InferShape(gs, &gin, &gaux, &gout);
	
	//cout << "xx" << endl;

	//d.InferShape(ds, &din, &daux, &dout);

	end.InferShape(ends, &endin, &endaux, &endout);
	
	vector <string> end_node_list = end.ListArguments();
	
	vector <string> g_node_list = g.ListArguments();

	vector <string> d_node_list = d.ListArguments();
	
	map_end["z"] = NDArray(Shape(nbatch, nnoise), ctx);

	map_end["lp"] = NDArray(Shape(nbatch), ctx);	
	
	map_end["inputs"] = NDArray(Shape(nbatch, size.height * size.width), ctx);

	init::init_weight_simple(end_node_list, endin, map_end, map_grad_end, ctx, init::init_mode::pretrained, pretrained);

	Executor * S = clip(Reshape(node["generated"], Shape(nbatch, size.height, size.width)), 0 + 1e-9, 1 - 1e-9).SimpleBind(ctx, map_end);
	
	Executor * G = g.SimpleBind(ctx, map_end, map_grad_end);
	
	Executor * D = d.SimpleBind(ctx, map_end, map_grad_end);
	
	
//	cout << "Create Optimizer " << endl;

 	Optimizer * Gadam = OptimizerRegistry::Find("adam");
 	
	Optimizer * Dadam = OptimizerRegistry::Find("adam");

	(*Gadam).SetParam("lr", 0.00005);
	
	(*Dadam).SetParam("lr", 0.00005);
	
	for (int i = 0; i != epoch; ++i){
		
		train_iter.Reset();
		
		while (train_iter.Next()){
      
			auto data_batch = train_iter.GetDataBatch();
		
			data_batch.data.CopyTo(&map_end["inputs"]);

			data_batch.label.CopyTo(&map_end["lp"]);
		
			//NDArray::SampleGaussian(0, 0.5, &map_end["z"]);
			
			uniform("z", &map_end["z"]);

			NDArray::WaitAll();
			
			//cout << "lp : " << map_end["lp"].Slice(0,1) << endl;

			D->Forward(true);
			
			D->Backward();
			
			
			for(int j = 0; j != (*D).arg_arrays.size(); ++j){

				string prefix = d_node_list[j].substr(0,4);

				if (prefix != "b_fc"){

					//cout << "D -> " << d_node_list[j] << " Weight " << (*D).arg_arrays[j].Reshape(Shape(-1)).Slice(100, 200) << endl << endl;

				//	cout << "D -> " << d_node_list[j] << " Gradient " << (*D).grad_arrays[j].Reshape(Shape(-1)).Slice(300, 400) << endl << endl;
					//cout << "D -> " << d_node_list[j] << " Gradient " << map_grad_end[d_node_list[j]].Reshape(Shape(-1)).Slice(500, 600) << endl << endl;

					
				}

				//cout << "qqq" << endl;

				if ((prefix == "w_fc") or (prefix == "b_fc")){
				
					(*Dadam).Update(j, (*D).arg_arrays[j], (*D).grad_arrays[j]);
					
					//cout << "Update : " << d_node_list[j] << endl;
				//cout << "D -> " << d_node_list[j] << " Weight " << (*D).arg_arrays[j].Reshape(Shape(-1)).Slice(0,1) << endl;
				
//					cout << "D -> " << d_node_list[j] << " Gradient " << (*D).grad_arrays[j].Reshape(Shape(-1)).Slice(0,1) << endl;
		
				}
			}
			

			
			(*G).Forward(true);

			(*G).Backward();


			for (int j = 0; j != (*G).arg_arrays.size(); ++j){
				
				string prefix = g_node_list[j].substr(0,4);
					
//				cout << "G -> " << g_node_list[j] <<  " Gradient " << (*G).grad_arrays[j].Reshape(Shape(-1)).Slice(0,1) << endl;
				
				if ((prefix == "w_de") or (prefix == "b_de")){ 		
					
					//cout << "Update : " << g_node_list[j] << endl;

					(*Gadam).Update(j, (*G).arg_arrays[j], (*G).grad_arrays[j]);
					
//					cout << "G -> " << g_node_list[j] << " Gradient " << (*G).grad_arrays[j].Reshape(Shape(-1)).Slice(0,20) << endl;
					

				}
			}			
					
		}	
		
		vector <NDArray> dout = (*D).outputs;
	
		vector <NDArray> gout = (*G).outputs;

		if (i % 10 == 0){

			cout << "Epoch : " << i << ", D Loss : " << dout[0] << ", G Loss : " << gout[0] << endl;
			
			(*S).Forward(false);
		
			vector <NDArray> sout = (*S).outputs;
			
			NDArray sslice = sout[0].Slice(0,1);

			fimage::saveb_1d("out_mnist/" + to_string(i), i, sslice, 255);

		
		}

		
	}
	
	delete G;

	delete D;

	delete S;

	MXNotifyShutdown();

};

CGAN_MNIST::~CGAN_MNIST(){
	
//	MXNotifyShutdown();

}

#endif
