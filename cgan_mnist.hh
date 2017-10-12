#ifndef CGAN_MNIST_HH
#define CGAN_MNIST_HH

#include <iostream>
#include <map>
#include <vector>
#include <mxnet-cpp/MxNetCpp.h>
#include <opencv2/opencv.hpp>
#include <boost/program_options.hpp>
#include "network.hh"
#include "config.h"
#include "cgan_mnist.h"
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
	
	cout << "Start Build" << endl;

	
	condition[0] = lp;

	condition[1] = z;

	/* 
	 *	condition => [nbatch, condition]
	 *	
	 *	condition_fc => map 2d "condition" [nbatch, condition] => [nbatch, channel = 128 , height, width] => deconvolution
	 */

	auto condition_concat = layer::concat("condition_concat", &condition, 1);

	node["generated"] = network::DEMLP(&condition_concat, &weight, &bias, size);
	
	/*
	 *	Convolution VGG
	 *	
	 *	Generated Image => flattened feature map + flattened condition => fullyconnected layer => real or fake
	 *
	 *
	 */

	node["decision_real"] = network::MLP(&inputs, &lp, nbatch, &weight, &bias, size); // presigmoid
	
	node["decision_fake"] = network::MLP(&node["generated"], &lp, nbatch, &weight, &bias, size); // presigmoid
	
};

inline Symbol CGAN_MNIST::Loss(){

	Symbol generate_loss = mean(loss::cross_entropy(node["decision_fake"],
				ones_like("generate_loss", node["decision_fake"])));
	
	Symbol real_decision_loss = mean(loss::cross_entropy(node["decision_real"],
				ones_like("real_loss", node["decision_real"])));
	
	Symbol fake_decision_loss = mean(loss::cross_entropy(node["decision_fake"],
				zeros_like("fake_loss", node["decision_fake"])));
		
	return MakeLoss("endloss", generate_loss + fake_decision_loss + real_decision_loss);
};

inline Symbol CGAN_MNIST::G_Loss(){

	/* let generated image can be considered a real image, so use ones_like, not zeros_like */

	auto generation_loss = mean(loss::cross_entropy(node["decision_fake"], ones_like("generate_loss", node["decision_fake"])));

//	loss_g = tf.reduce_mean(-tf.log(D2))
	//return MakeLoss("G_Loss", generation_loss);
	//
	return MakeLoss("G_Loss", mean(0 - log(node["decision_fake"])));
}


inline Symbol CGAN_MNIST::D_Loss(){	

	//Symbol real_decision_loss = mean(loss::cross_entropy(node["decision_real"],
	//			ones_like("real_loss", node["decision_real"])));
	
	//Symbol fake_decision_loss = mean(loss::cross_entropy(node["decision_fake"],
	//			zeros_like("fake_loss", node["decision_fake"])));

	return MakeLoss("D_Loss", mean(0 - log(node["decision_real"]) - log(1 - node["decision_fake"])));
//	loss_g = tf.reduce_mean(-tf.log(D2))
//	return MakeLoss("D_Loss", real_decision_loss + fake_decision_loss);
}


inline void CGAN_MNIST::train(int epoch){
	
	Context ctx(device, 0);
	
	data::db dataset(CGAN_LP::sdataset, CGAN_LP::slist, CGAN_LP::label, CGAN_LP::size, CGAN_LP::nbatch, true, data::MODE::generation);

	dataset.next();
	
	
	map <string, map <string, vmx_shape>> infered_shape;

	map <string, map <string, mx_shape>> arg_shape;
	
	map <string, NDArray> nd, grad;
	
	arg_shape["end"]["inputs"] = {nbatch, size.height * size.width};

	arg_shape["end"]["lp"] = {nbatch};
	
	arg_shape["end"]["z"] = {nbatch, nnoise};

	arg_shape["g"]["lp"] = {nbatch};
	
	arg_shape["g"]["z"] = {nbatch, nnoise};
	
	/* setup ndarray */

	//ndarg["inputs"] = fimage::MatVector_to_NDArray(dataset.inputs, ctx);

	//cout << "Inputs : " << Shape(ndarg["inputs"].GetShape()) << endl;
	
	//ndarg["z"] = NDArray(Shape(nbatch, nnoise), ctx);

	//ndarg["lp"] = lp::generate_one_hot(Shape(nbatch, nobject, nclass), label, dataset.target, ctx);
	

	//Xavier uniform(0, 1);

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

	
	Symbol g = G_Loss();//node["decision_real"];

	Symbol d = D_Loss();

	Symbol end = g + d;

	//node["generated"].InferShape(arg_shape["g"], , &gen_aux, &gen_out);

	//cout << "Generated Image : " << Shape(gen_out[0]) << endl;
	
	g.InferShape(arg_shape["g"], &infered_shape["g"]["in"], &infered_shape["g"]["aux"], &infered_shape["g"]["out"]);
	
	d.InferShape(arg_shape["d"], &infered_shape["d"]["in"], &infered_shape["d"]["aux"], &infered_shape["d"]["out"]);

	end.InferShape(arg_shape["end"], &infered_shape["end"]["in"], &infered_shape["end"]["aux"], &infered_shape["end"]["out"]);
	
	vector <string> end_node_list = end.ListArguments();
	
	vector <string> g_node_list = g.ListArguments();

	vector <string> d_node_list = d.ListArguments();
	
	
	//map <string, Shape> node_shape = merge_uint_vector(node_name, in_shape);
	
	nd = vector <NDArray> (end_node_list.size());

	grad = vector <NDArray> (end_node_list.size());

	
	nd["z"] = NDArray(Shape(nbatch, nnoise), ctx);

	nd["lp"] = NDArray(Shape(nbatch), ctx);	
	
	nd["input"] = NDArray(Shape(nbatch, size.height * size.width), ctx);

	/* setup weight */

	/* Deprecated */
	
	

	init::init_weight(end_node_list, endin, nd_end, grad_end, ctx, init::init_mode::pretrained, pretrained);

	auto wrt_g = init::wrt(vector <char *> {"de"}, g_node_list);

	auto wrt_d = init::wrt(vector <char *> {"fc"}, d_node_list);

	/* setup executor */

	//Executor * S = Reshape(node["generated"], Shape(nbatch, size.height, size.width)).Bind(ctx, nd_g, grad_g, wrt_g, vector <NDArray> ());
	
	//Executor * D = d.Bind(ctx, nd_d, grad_d, wrt_d, vector <NDArray> ());
	
	//Executor * G = g.Bind(ctx, nd_g, grad_g, wrt_g, vector <NDArray> ());
	
	Executor * S = Reshape(node["generated"], Shape(nbatch, size.height, size.width)).SimpleBind(ctx, map_end);
	
	for (auto &i : map_end)

		cout << i.first << " : " << Shape(i.second.GetShape()) << endl;

	Executor * D = d.SimpleBind(ctx, map_end, map_grad_d);
	
	Executor * G = g.SimpleBind(ctx, map_end, map_grad_g);

	
	//auto reconstruction_loss = mean(loss::cross_entropy(node["generated"], inputs));
	
	cout << "Create Optimizer " << endl;

 	Optimizer * Gadam = OptimizerRegistry::Find("adam");
 	
	Optimizer * Dadam = OptimizerRegistry::Find("adam");

	(*Gadam).SetParam("lr", 0.00005);
	
	(*Dadam).SetParam("lr", 0.00005);

	int step = 0;
	
	for (int i = 0; i != epoch; ++i){
		
		train_iter.Reset();
		
		while (train_iter.Next()){
      
			auto data_batch = train_iter.GetDataBatch();
			
			step += 1;

			data_batch.data.CopyTo(&nd_end[input_end]);
			//data_batch.data.CopyTo(&map_end["inputs"]);

			data_batch.label.CopyTo(&nd_end[endlp]);
      		
			//cout << data_batch.label << endl;

			NDArray::WaitAll();
		
			nd_d[input_d] = nd_end[input_end];

			map_end["inputs"] = nd_end[input_end];
			
			nd_g[glp] = nd_end[endlp];
			
			nd_d[dlp] = nd_end[endlp];
			
			map_end["lp"] = nd_end[endlp];
			
			//uniform("z", &nd_end[endz]);
		
			NDArray::SampleGaussian(0, 1, &nd_end[endz]);
			
			nd_g[gz] = nd_end[endz];		
		
			nd_d[dz] = nd_end[endz];

			map_end["z"] = nd_end[endz];
			
			for (int iter_d = 0; iter_d != 1; ++iter_d){

				(*D).Forward(true);
				
				(*D).Backward();
				
				//for (int j = 0; j != grad_d.size(); ++j){
				
				for(int j = 0; j != (*D).arg_arrays.size(); ++j){

					string prefix = d_node_list[j].substr(0,2);

					/*if (d_node_list[j] == "wfc1"){
						//cout << "D : " << d_node_list[j] << " weight : " << &(*D).arg_arrays[j] << endl;	
						//cout << "D : " << d_node_list[j] << " grad : " << &(*D).grad_arrays[j] << endl << endl;
					}*/
		
					cout << "D : " <<  d_node_list[j] << " : " << (*D).arg_arrays[j].Reshape(Shape(-1)).Slice(0, 1) << endl;

					if ((prefix == "wf") or (prefix == "bf")){
						
						(*Dadam).Update(j, (*D).arg_arrays[j], (*D).grad_arrays[j]);
				
						//cout << map_d[d_node_list[j]] << endl;
					}
				}	
			}

			

			for(int ig = 0; ig != g_node_list.size(); ++ig){

				auto idx = find(d_node_list.begin(), d_node_list.end(), g_node_list[ig]) - d_node_list.begin();
				
				//nd_g[ig] = nd_d[idx];

				/*if (g_node_list[ig] == "wdefc1"){

					cout << "========================" << endl;
					
					cout << "D -> G: " << g_node_list[ig] << " weight : " << nd_g[ig].Reshape(Shape(-1)).Slice(0,5) << endl;
							
					cout << "D -> G: " << g_node_list[ig] << " weight : " << nd_d[idx].Reshape(Shape(-1)).Slice(0,5) << endl;
					
					cout << "D -> G: " << g_node_list[ig] << " weight : " << (*D).arg_arrays[idx].Reshape(Shape(-1)).Slice(0,5)  << endl;
							
					cout << "D -> G: " << g_node_list[ig] << " weight : " << (*G).arg_arrays[ig].Reshape(Shape(-1)).Slice(0,5)  << endl << endl;
					
					cout << "D -> G: " << g_node_list[ig] << " gradient : " << grad_g[ig].Reshape(Shape(-1)).Slice(0,5) << endl;
							
					cout << "D -> G: " << g_node_list[ig] << " gradient : " << grad_d[idx].Reshape(Shape(-1)).Slice(0,5) << endl;
					
					cout << "D -> G: " << g_node_list[ig] << " gradient : " << (*D).grad_arrays[idx].Reshape(Shape(-1)).Slice(0,5)  << endl;
							
					cout << "D -> G: " << g_node_list[ig] << " gradient : " << (*G).grad_arrays[ig].Reshape(Shape(-1)).Slice(0,5)  << endl << endl;

					cout << endl << "========================" << endl;
					
				}*/
			}

			for (int iter_g = 0; iter_g != 1; ++iter_g){
			
				(*G).Forward(true);

				(*G).Backward();	
			
				for (int j = 0; j != (*G).arg_arrays.size(); ++j){
				//for (int j = 0; j != grad_g.size(); ++j){
						
					/*if (g_node_list[j] == "wfc1"){
				//		cout << "G : " << g_node_list[j] << " weight : " << &(*G).arg_arrays[j] << endl;
						
				//		cout << "G : " << g_node_list[j] << " grad : " << &(*G).grad_arrays[j] << endl << endl;;
					}*/	
					string prefix = g_node_list[j].substr(0,2);
						
					if ((prefix == "wd") or (prefix == "bd")){ 		

						(*Gadam).Update(j, (*G).arg_arrays[j], (*G).grad_arrays[j]);

					}
				}			
			}

			/*for(int id = 0; id != d_node_list.size(); ++id){
				
				if (d_node_list[id] != "inputs"){

				auto idx = find(g_node_list.begin(), g_node_list.end(), d_node_list[id]) - g_node_list.begin();
				
				//nd_d[id] = nd_g[idx];

				}
			}*/

			vector <NDArray> dout = (*D).outputs;
		
			vector <NDArray> gout = (*G).outputs;
			
			if (step % 1000 == 0)

				cout << "Epoch : " << i << ", D Loss : " << dout[0] << ", G Loss : " << gout[0] << endl;
			

			
			if (step % 10000 == 0 and step != 0){
			
				NDArray::SampleGaussian(0, 1, &nd_end[endz]);

				(*S).Forward(false);
			
				vector <NDArray> sout = (*S).outputs;
				
				NDArray sslice = sout[0].Slice(0,1);

				fimage::saveb_1d("out_mnist/" + to_string(i), i, sslice, 255);

			}
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
