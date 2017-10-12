#ifndef DATA_HH
#define DATA_HH

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

#include <rapidxml/rapidxml.hpp>
#include <rapidxml/rapidxml_utils.hpp>
#include <rapidxml/rapidxml_print.hpp>

#include <algorithm>
#include <lmdb++.h>
#include <lmdb.h>
#include <unistd.h>
#include <leveldb/db.h>
#include <assert.h>
#include <flt.h>
#include "annotationdb.pb.h"
#include "augmentation.hh"


using namespace std;



data::db::db(string _sdataset,
		string strain,
		vector <string> _label,
		cv::Size _size,
		int _batch_size = 1,
		bool debug = false,
		data::MODE _mode = data::MODE::detection){

	sdataset = _sdataset;

	width = _size.width;

	height = _size.height;

	batch_size = _batch_size;

	label = _label;

	DEBUG = debug;

	mode = _mode;

	basedir = string("Data") + "/" + sdataset + "/";

	annotationdir = basedir + "Annotations" + "/";

	n_class = label.size();

	cout << "111" << endl;
	
	if (!boost::filesystem::exists(sdataset)){
				
		cout << "Dataset : " << sdataset << endl;

		printf("[*] Dataset %s is not exist ...\n", sdataset);

		printf("[*] Creating Dataset ...\n");

		options.create_if_missing = true;
		
		leveldb::Status status = leveldb::DB::Open(options, sdataset, &env);

		assert(status.ok());
		
		generate(strain);
		
	}

	else{

		leveldb::Status status = leveldb::DB::Open(options, sdataset, &env);

		assert(status.ok());

	}

		leveldb::Status s;
		
		string s_image_entry, s_proposal_entry;
	cout << "22" << endl;
		if (mode == MODE::detection){
			cout << "uu" << endl;
			s = env->Get(leveldb::ReadOptions(), "Image Entry", &s_image_entry);

			assert(s.ok());

			s = env->Get(leveldb::ReadOptions(), "Proposal Entry", &s_proposal_entry);

			assert(s.ok());
		}

		else if (mode == MODE::generation){

			s = env->Get(leveldb::ReadOptions(), "Image Entry", &s_image_entry);
			
			cout << "s Image entry : " << s_image_entry << endl;
			assert(s.ok());

			s = env->Get(leveldb::ReadOptions(), "Image Entry", &s_proposal_entry);
			
			cout << "s Proposal Entry : " << s_proposal_entry << endl;
			assert(s.ok());

		}

		n_proposal_entry = stoi(s_proposal_entry);

		n_image_entry = stoi(s_image_entry);

			
		
		for(int i = 0; i != n_proposal_entry; ++i){

			proposals_batch.push_back(i); // batch index start from 0, because it is index, not key

			proposals_key.push_back(i); // image entry start from 1, it is key
		}

		shuffle();
	
}



inline int data::db::generate(string list){
	
	n_image_entry = 0;

	n_proposal_entry = 0;

	/*

	Generate LMDB , by Iterate along the image list

	*/

	//lmdb::txn wtxn = lmdb::txn::begin(env);

	//lmdb::dbi wdbi = lmdb::dbi::open(wtxn);
	
	//wdbi.put(wtxn, "Dataset", "Alphabet");
	
	
	leveldb::Status s;
	
	s = env->Put(leveldb::WriteOptions(), "Dataset", sdataset);
	
	assert(s.ok());

	flt::ffile::fiterator iter(basedir + list);

	//cout << "after iter " << endl;

	while(iter.next()){

		string path = basedir + "Images" + "/"+ iter.line;

		cout << "[Generate] Image : " << path + ".jpg -> ";

		//cv::Mat image = imread(path + ".jpg", cv::IMREAD_COLOR);

		//int w = image.size().width;

		//int h = image.size().height;

		if (mode == MODE::generation)
			
			db::merge_generative(iter.line);

		else if(mode == MODE::detection)

			db::merge(iter.line);

		n_image_entry += 1;
	}
	
	//wdbi.put(wtxn, "Image Entry", to_string(n_image_entry).c_str());

	
	s = env->Put(leveldb::WriteOptions(), "Image Entry", to_string(n_image_entry));
	
	assert(s.ok());

	s = env->Put(leveldb::WriteOptions(), "Proposal Entry", to_string(n_proposal_entry));
	
	assert(s.ok());
	//wtxn.commit();

  	//wtxn.abort();
}

inline void data::db::generate_label_array(int *v, int index){

	for(int i = 0; i != n_class; i++)

		v[i] = 0; // set all values to 0

	v[index] = 1; // set index to 1
}


inline void data::db::generate_proposal_list(){

	for(int i = 0; i != n_proposal_entry; i++)

		_proposal_list.push_back(i);

	//db::shuffle();

}

inline int data::db::shuffle(){

	unsigned seed = (unsigned)time(NULL); // 取得時間序列

	//srand(seed);

	random_shuffle(proposals_key.begin(), proposals_key.end());

}

inline int data::db::merge(string filename){

	/*

	Merge Annotation files and Image Path to LMDB File

	*/

	//rapidxml::file <> xmlFile("xmlpath");

	//cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;

	string path = annotationdir + filename + ".xml";

	//cout << "Path : " << path << endl;

	rapidxml::xml_node <> * root = flt::fxml::load_xml(path);

	//cout << "Loaded xml" << endl;

	rapidxml::xml_node <> * bbox;

	rapidxml::xml_node <> * size = flt::fxml::find_child_node(root, "size");

	rapidxml::xml_node <> * xmlfilename = flt::fxml::find_child_node(root, "filename");

	int image_width = stoi(flt::fxml::find_child_node(size, "width")->value());

	int image_height = stoi(flt::fxml::find_child_node(size, "height")->value());

	int xmin, ymin, xmax, ymax;

	//cout << "Loaded ann" << endl;

	string c;

	string encode;




	rapidxml::xml_node <> * object = flt::fxml::find_child_node(root, "object");

	while (true){

		if (object == nullptr)

			break;
		else

			cout << "object : " << object->name() << endl;

	//	cout << "Find child node ..." << endl;

		//cout << "filename : " << xmlfilename->value() << endl;


		c = flt::fxml::find_child_node(object, "name")->value();

	//	cout << "c : " << c << endl;

		bbox = flt::fxml::find_child_node(object, "bndbox");



		//cout << "Find bndbox" << endl;

		xmin = stoi(flt::fxml::find_child_node(bbox, "xmin")->value());

		ymin = stoi(flt::fxml::find_child_node(bbox, "ymin")->value());

		xmax = stoi(flt::fxml::find_child_node(bbox, "xmax")->value());

		ymax = stoi(flt::fxml::find_child_node(bbox, "ymax")->value());

		//c = flt::fxml::find_child_node(object, "name")->value();

		//cout << "c : " << c << endl;





		//cout << "annotationdb ... " << endl;

		annotationdb::proposal proposal;

		proposal.set_xmin(xmin);

		proposal.set_ymin(ymin);

		proposal.set_w(xmax - xmin);

		proposal.set_h(ymax - ymin);

		proposal.set_c(c);

		proposal.set_entry(string(filename + ".jpg"));

		proposal.SerializeToString(&encode);
		
		cout << "[Merge][PUT] " <<  to_string(n_proposal_entry) << endl;
		
		env->Put(leveldb::WriteOptions(), to_string(n_proposal_entry), encode);

		//cout << "wdbi put" << endl;

		n_proposal_entry += 1;

		//cout << "n_proposal_entry += 1" << endl;

		object = flt::fxml::find_sibling_node(object, "object");

	}
 
	//delete root; delete bbox; delete object; delete size;

	cout << "exit merge ..." << endl;
}


inline int data::db::merge_generative(string filename){
	
	/*

	Merge Annotation files and Image Path to LMDB File

	*/

	//rapidxml::file <> xmlFile("xmlpath");

	//cout << "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;

	string path = annotationdir + filename + ".xml";

	//cout << "Path : " << path << endl;

	rapidxml::xml_node <> * root = flt::fxml::load_xml(path);

	//cout << "Loaded xml" << endl;

	rapidxml::xml_node <> * bbox;

	rapidxml::xml_node <> * size = flt::fxml::find_child_node(root, "size");

	rapidxml::xml_node <> * xmlfilename = flt::fxml::find_child_node(root, "filename");

	int image_width = stoi(flt::fxml::find_child_node(size, "width")->value());

	int image_height = stoi(flt::fxml::find_child_node(size, "height")->value());

	int xmin, ymin, xmax, ymax;

	//cout << "Loaded ann" << endl;

	string c;

	string encode;
	
	annotationdb::image protobuf;
	
	vector <string> class_vector;

	rapidxml::xml_node <> * object = flt::fxml::find_child_node(root, "object");

	protobuf.set_entry(string(filename + ".jpg"));
	
	cout << "N [" << n_image_entry << "]";
	cout << "Merge [";
	
	while (true){

		if (object == nullptr)

			break;

		c = flt::fxml::find_child_node(object, "name")->value();
		
		cout << c;
		//bbox = flt::fxml::find_child_node(object, "bndbox");

		//xmin = stoi(flt::fxml::find_child_node(bbox, "xmin")->value());

		//ymin = stoi(flt::fxml::find_child_node(bbox, "ymin")->value());

		//xmax = stoi(flt::fxml::find_child_node(bbox, "xmax")->value());

		//ymax = stoi(flt::fxml::find_child_node(bbox, "ymax")->value());
		
		protobuf.add_object(c);

		n_proposal_entry += 1;

		object = flt::fxml::find_sibling_node(object, "object");

	}
	
	cout << "] " << endl;

	protobuf.SerializeToString(&encode);
	
	env->Put(leveldb::WriteOptions(), to_string(n_image_entry), encode);
	
	//delete root; delete bbox; delete object; delete size;

	//cout << "exit merge ..." << endl;
}

inline int data::db::next(){

	/*

	return next proposal

	protobuf.set_image_entry(); // index to image
	*/


	//cout << "Front of Next ..." << endl;
	
	//cout << endl << endl << "=================" << endl << endl;
	
	//cout << "	NEXT " << endl;


	//cout << endl << endl << "=================" << endl << endl;
	
	//cout << "step : " << step << endl;

	//cout << "batch_size : " << batch_size << endl;

	//cout << "n_proposal_entry : " << n_proposal_entry << endl;
	
//	cout << "in function next ... " << endl;	
	
	//cout << n_proposal_entry << endl;

	_begin = step * batch_size % n_proposal_entry;
	
//	cout << "[Next] _begin" << endl;

	_end = (step + 1) * batch_size % n_proposal_entry;

	//cout << "[Next] _end" << endl;

	//cout << "Before if ... " << endl;



	if((_begin + batch_size) > n_proposal_entry){

		//cout << "Before if (1) ..." << endl;

		batch = flt::fvector::concat(
				flt::fvector::slice(proposals_batch, _begin, n_proposal_entry),
				flt::fvector::slice(proposals_batch, 0, _end));

		//cout << "After if (1) ...";
	}
	else if((_begin + batch_size) == n_proposal_entry){

		//cout << "Before if (2) ..." << endl;

		batch = flt::fvector::slice(proposals_batch, _begin, n_proposal_entry);

		//cout << "After if (2) ...";
	}

	else{

		//cout << "Before if (3) ..." << endl;

		//cout << "begin : " << _begin << endl;

		//cout << "end : " << _end << endl;

		batch = flt::fvector::slice(proposals_batch, _begin, _end);

		//cout << "After if (3) ..." << endl;
	}

	//cout << "After if ..." << endl;

	step += 1;

	shuffle_counter -= batch_size;

	if (shuffle_counter <= 0){

		flt::fdebug::log("Shuffle Proposal Entry ...", DEBUG);

		shuffle();

		shuffle_counter = n_proposal_entry;
	}


	//cout << "Load Batch ... " << endl;
	
	if (mode == MODE::generation)

		data::db::load_generative(batch);

	else if(mode == MODE::detection)

		data::db::load_detection(batch);

	//cout << "end of next" << endl;
}

inline int data::db::show(){

	/*

	Debug function , to see if the bounding box loading work normally

	*/

	cv::Mat image;

	//cv::Scalar color = (255,255,0);

	inputs[0].copyTo(image);


	int x = target[0].x;

	int y = target[0].y;

	int w = target[0].w;

	int h = target[0].h;

	cv::rectangle(image, cv::Point(x, y), cv::Point(x + w, y + h), cv::Scalar(255,255,0), 2);

	cv::imshow("Image", image);

	cv::waitKey(0);

}

inline int data::db::load_detection(vector <int> batch){

	/*

	Load proposal from LMDB file and do Data Augmentation on the fly

	load proposal by given shuffled list

	batch : continuous list

	_proposal list : random shuffled list

	*/

	//cout << "clear ..." << endl;
	
	cout << "load detection .. " << endl;
	inputs.clear();

	target.clear();

	//cout << "[inputs] cleared" << endl;

	//cout << "[targets] cleared" << endl;

	string encoded;

	if (batch_size != batch.size())

		cout << "batch size is not correct";

	//cout << "After check size " << endl;

	//cout << "Batch : " << batch.size() << endl;

	for(int i = 0; i != batch.size(); i++){

		//cout << "Before get ..." << endl;

		//printf("Batch [%d] : %d\n", i, batch[i]);

		//cout << _proposal_list[batch[i]] << endl;

		//cout << to_string(_proposal_list[batch[i]]).c_str() << endl;

		//(*dbi).get(*rtxn, to_string(_proposal_list[batch[i]]).c_str(), encoded);

		//cout

		//string sss = ;


		//lmdb::val key {to_string(0).c_str()};

	//	cout << "After create key ..." << endl;
		
		string value;

		env->Get(leveldb::ReadOptions(), to_string(proposals_batch[batch[i]]), &value);

		annotationdb::proposal decoded;

		decoded.ParseFromString(value);

		//cout << "After ParseFromString ... " << endl;

		annotation ann;

		ann.x = decoded.xmin();

		ann.y = decoded.ymin();

		ann.w = decoded.w();

		ann.h = decoded.h();

		//cout << "ParseFromString ... Get Value" << endl;

		//cout << "decode class : " << decoded.c() << endl;

		int c = _get_index_by_class(decoded.c());

		//cout << "Get c ..." << endl;

		//cout << "C : " << c << endl;

		if (c == -1)

			throw "given class does not exist";

		//cout << "After throw ..." << endl;

		ann.i = new int [n_class - 1]; // exclude background

		//cout << "Generate label array ..." << endl;

		generate_label_array(ann.i, c);

		//cout << "ParseFromString ... Get Value, After" << endl;

		//cout << "c : " << c << endl;


		ann.c = c;

		/*

		int x = decoded.x();

		int y = decoded.y();

		int w = decoded.w();

		int h = decoded.h();

		string c = decoded.c();

		*/

		//string e = decoded.entry();


		//cout << "decode entry : " << decoded.entry().c_str() << endl;


		string path = basedir + string("Images") + string("/") + decoded.entry();

		//cout << "PATH : " << path << endl;
		//string path = string("%s/Images%d.jpg", basedir.c_str(), 0);

		//cout << string("Image Path : %s", path.c_str()) << endl;

		//cout << "xxxxxxx" << endl;

		flt::fdebug::log(string("Image Path : " + path), DEBUG);

		//cout << path << endl;

		cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

		cv::Mat rimage(height, width, 3);

		cv::resize(image, rimage, cv::Size(height, width));

		augmentation::method method;

		ulBox ulbox;

		ulbox.xmin = ann.x;
		ulbox.ymin = ann.y;
		ulbox.w = ann.w;
		ulbox.h = ann.h;

		//if(!config.au)

		//ulbox = augmentation::augmentation(&rimage, ulbox, method);

		inputs.push_back(rimage);

		target.push_back(ann);
		
		cout << "After Push " << endl;
		/*
		cout << "load class : ";

		for(int i = 0; i != n_class; i++)

			cout << ann.i[i] << ", ";

		cout << endl;

		*/

	}

}


inline int data::db::load_generative(vector <int> minibatch){

	/*

	Load proposal from LMDB file and do Data Augmentation on the fly

	load proposal by given shuffled list

	batch : continuous list

	_proposal list : random shuffled list

	*/
	
	//cout << endl << endl;

	//cout << "load Generative" << endl << endl;
	//cout << "before load generative" << endl;

	inputs.clear();

	target.clear();

	string encoded;

	if (batch_size != minibatch.size())

		cout << "batch size is not correct";

	for(int i = 0; i != minibatch.size(); i++){
		
		/*
		 *	proposals key is a randomize real key vector
		 *	
		 *	minibatch[i] : i = batch size, minibatch contains a continueous value 
		 *
		 *	for exmaple [ 25 - 70 ]
		 *
		 *	but 25 - 70 may map to different real key in "proposals key" each iterations
		 *
		 *
		 */
		
		//cout << "Key : " << to_string(proposals_batch[minibatch[i]]) << endl;
		
		string value;

		env->Get(leveldb::ReadOptions(), to_string(proposals_key[minibatch[i]]), &value);

		annotationdb::image deprotobuf;
		
		deprotobuf.ParseFromString(value);
		
		//cout << deprotobuf.entry() << endl;

		annotation ann;
		
		//cout << "Number of Object : " << deprotobuf.object_size() << endl;

		vector <string> class_vector(deprotobuf.object_size());
		
		for (int i = 0; i != deprotobuf.object_size(); ++i)

			class_vector[i] = deprotobuf.object(i);
		
		/* use only class vector for generation */

		/* add bounding box location in the further future */

		ann.vs = class_vector; 

		/*cout << "c[0] : "<< class_vector[0] << endl;
		cout << "c[1] : "<< class_vector[1] << endl;
		cout << "c[2] : "<< class_vector[2] << endl;
		cout << "c[3] : "<< class_vector[3] << endl;
			
		cout << "Deprotobuf : " << deprotobuf.entry() << endl;
		*/
		string path = basedir + string("Images") + string("/") + deprotobuf.entry();

		//flt::fdebug::log(string("Image Path : " + path), DEBUG);

		//cout << "Path : " << path << endl;

		cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);

		cv::Mat rimage(height, width, CV_8UC3);

		cv::Mat frimage(height, width, CV_32FC3);

		cv::resize(image, rimage, cv::Size(height, width));
		
		rimage.convertTo(frimage, CV_32FC3);
		
		augmentation::method method;

		ulBox ulbox;

		ulbox.xmin = ann.x;
		ulbox.ymin = ann.y;
		ulbox.w = ann.w;
		ulbox.h = ann.h;

		//if(!config.au)

		//ulbox = augmentation::augmentation(&rimage, ulbox, method);

		inputs.push_back(frimage);

		target.push_back(ann);
		
		//cout << "After Push ... " << endl;
		/*
		cout << "load class : ";

		for(int i = 0; i != n_class; i++)

			cout << ann.i[i] << ", ";

		cout << endl;

		*/
		//cout << "AFter Push" << endl;
	}
	
	//cout << "yyy" << endl;
}

inline int data::db::_get_index_by_class(string c){

	/*

	return the index of the given class

	*/

	//cout << "label size : " << label.size() << endl;
	
	cout << "sss" << endl;
	if(!label.size())

		flt::fdebug::error(string("Label size is not correct : %d", label.size()));

	//cout << "middle ..." << endl;


	for(int i = 0; i != label.size(); i++){

		//cout << "C : " << c << endl;

		if(c == label[i]) // class 0 is background;


			if(!i)

				cout << "class index error ..." << endl;

			else

				return i - 1;

		//else

			//cout << "i : " << i << endl;
	}

//	cout << "get label index .. exit" << endl;

	return -1;
}

inline void data::MatVector_to_NDArray(vector <cv::Mat> &v, NDArray &n){

	int h = v[0].size().height;

	int w = v[0].size().width;

	int channel = v[0].channels();

	int b = v.size();

	float * ftotal = new float [b * h * w * channel];

	float * fim;

	int size = h * w * channel;

	//cout << "before for loop" << endl;

	for(int i = 0; i != b; i++){

		cv::Mat temp(h, w, CV_32FC3);

		v[i].convertTo(temp, CV_32FC3, 1 / 255.0);

		fim = (float *)temp.data;

		copy(fim, fim + size, ftotal + i * size);

	}

	//cout << "after for loop" << endl;

	n.SyncCopyFromCPU(ftotal, b * size);

	NDArray::WaitAll();

	//delete fim;

	delete ftotal;

	//cout << "before return ..." << endl;

}


inline NDArray data::MatVector_to_NDArray(vector <cv::Mat> v, Context device){

	int h = v[0].size().height;

	int w = v[0].size().width;

	int channel = v[0].channels();

	int b = v.size();

	float * ftotal = new float [b * h * w * channel];

	int size = h * w * channel;

	Shape sim(h, w, channel);

	Shape stotal(b, h, w, channel);

	for(int i = 0; i != b; i++){

		cv::Mat temp(h, w, CV_32FC3);

		v[i].convertTo(temp, CV_32FC3, 1 / 255.0);

		float * fim = (float *)temp.data;

		copy(fim, fim + size, ftotal + i * size);

	}

	//Shape stotal(b, h, w, channel);

	NDArray ndtotal(stotal, device);

	ndtotal.SyncCopyFromCPU(ftotal, b * size);

	NDArray::WaitAll();

	return ndtotal;

}

inline vector <cv::Mat> NDArray_to_MatVector(NDArray n){

	vector <mx_uint> shape = n.GetShape();

	int b = shape[0];

	int h = shape[1];

	int w = shape[2];

	int channel = shape[3];

	vector <cv::Mat> v(b);

	float * ftotal = new float [b * h * w * channel];

	int size = h * w * channel;

	Shape sim(h, w, channel);

	Shape stotal(b, h, w, channel);

	n.SyncCopyToCPU(ftotal, b * size);

	NDArray::WaitAll();

	for(int i = 0; i != b; i++){

		float * fim = new float [size];

		copy(ftotal + i * size, ftotal + (i + 1) * size, fim);

		v[i] = cv::Mat(h, w, CV_32FC3, fim);

	}

	return v;
}

/*
int main(){

	//db database("Alphabet", "train_multi.txt", 300, 300, 10);

	//ppp();

	SIZE size;

	srand(getpid());

	size.w = 300;

	size.h = 300;

	vector <string> label {


		"Background",

		"A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
		"K", "L", "M", "N", "P", "Q", "R", "S", "T", "U",
		"V", "W", "X", "Y", "Z",

		"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"

	};


	db database("Alphabet", "train_multi.txt", label, size, 32, true);



	database.next();

	flt::debug::log("Done ... ", true);

	database.show();

	return 0;

}

*/


#endif
