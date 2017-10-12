#ifndef AUGMENTATION_HH
#define AUGMENTATION_HH
#include <iostream>
#include <opencv2/opencv.hpp>

#include <flt.h>
#include "augmentation.hh"
#include "augmentation_mapping.hh"
#include "data.hh"


using namespace std;


float random_choice(vector <float>);

namespace augmentation{



	data::ulBox flip(cv::Mat *in, data::ulBox ulbox){

		ulbox.xmin = (*in).size().width - ulbox.xmin - ulbox.w;

		cv::flip(*in, *in, 1);

		return ulbox;
	}

	int relsolution(cv::Mat *in, vector <float> v){

		float r = random_choice(v);

		cv::resize(*in, *in,

			cv::Size(

				(int)((float)(*in).cols * r),
				(int)((float)(*in).rows * r)

			)
		);
	}

	int brightness(cv::Mat *in, vector <float> v){

		float r = random_choice(v);

		int w = (*in).size().width;

		int h = (*in).size().height;

		for(int j = 0; j != h; j++)

			for(int i = 0; i != w; i++){

				flt::fcv::set(in, i, j, HSV::V, (int)((float)flt::fcv::get(in, i, j, HSV::V) * r));

			}

	}

	int saturation(cv::Mat *in, vector <float> v){

		float r = random_choice(v);

		int w = (*in).size().width;

		int h = (*in).size().height;

		for(int j = 0; j != h; j++)

			for(int i = 0; i != w; i++){

				flt::fcv::set(in, i, j, HSV::S, (int)((float)flt::fcv::get(in, i, j, HSV::S) * r));

			}

	}

	int hue(cv::Mat *in, vector <float> v){

		float r = random_choice(v);

		//cout << "r : " << r << endl;

		int w = (*in).size().width;

		int h = (*in).size().height;

		for(int j = 0; j != h; j++)

			for(int i = 0; i != w; i++){

				flt::fcv::set(in, i, j, HSV::H, (int)((float)flt::fcv::get(in, i, j, HSV::H) * r));

			}

	}

	int gaussian_blur(cv::Mat *in){


	}

	data::ulBox random_proposal(cv::Mat *in, data::ulBox ulbox, cv::Size strain){

		/*

		s: size
		n: new
		c: center
		b: border
		ul: upper left


		(*in): original image

		proposal: cropped around the center of bounding box



		*/

		cv::Size sin;

		sin.width = (*in).size().width;

		sin.height = (*in).size().height;


		cout << "sin w : " << sin.width << endl;

		cout << "sin h : " << sin.height << endl;

		cout << "ulbox xmin : " << ulbox.xmin << endl;

		cout << "ulbox ymin : " << ulbox.ymin << endl;

		cout << "ulbox w : " << ulbox.w << endl;

		cout << "ulbox h : " << ulbox.h << endl;



		data::cBox cbox = mapping::upper_left_to_center(ulbox);

		data::bBox bbox = mapping::upper_left_to_border(ulbox);

		data::RATIO scale;

		data::RATIO scale_max;


		scale_max.w = (float)sin.width / cbox.w;

		scale_max.h = (float)sin.height / cbox.h;

		try {

			vector <float> vscale_w = flt::fvector::arange(1.1, scale_max.w, 0.5);

			if(!vscale_w.size())

				throw string("no item in the vector");

			scale.w = random_choice(vscale_w);

			vector <float> vscale_h = flt::fvector::arange(1.1, scale_max.h, 0.5);

			if(!vscale_h.size())

				throw string("no item in the vector");

			scale.h = random_choice(vscale_h);

			//scale.h = random_choice(tool::xvector::arange(1.1, scale_max.h, 0.5));

		} catch (string e) {

			scale.w = scale_max.w;

			scale.h = scale_max.h;
		}


		cv::Size sproposal; // cropped from original ;

		data::cBox cproposal;




		sproposal.width = cbox.w * scale.w;

		sproposal.height = cbox.h * scale.h;

		cproposal.x = cbox.x;

		cproposal.y = cbox.y;

		cproposal.w = sproposal.width;

		cproposal.h = sproposal.height;

		data::bBox bproposal = mapping::center_to_border(cproposal);

		cout << "cproposal x : " << cproposal.x << endl;

		cout << "cproposal y : " << cproposal.y << endl;

		cout << "cproposal w : " << cproposal.w << endl;

		cout << "cproposal h : " << cproposal.h << endl;


		//mapping::draw(in, bproposal, "cproposal");



		data::RATIO ratio_min, ratio_max;

		ratio_min.w = cbox.w * 0.5 / sproposal.width;

		ratio_max.w = 1 - cbox.w * 0.5 / sproposal.width;

		ratio_min.h = cbox.h * 0.5 / sproposal.height;

		ratio_max.h = 1 - cbox.h * 0.5 / sproposal.height;

		data::RATIO ratio;



		vector <float> vratio;

		vratio = flt::fvector::arange(ratio_min.w + 0.05, ratio_max.w - 0.05, 0.1);

		if(!vratio.size())

			ratio.w = 0.5;

		else

			ratio.w = random_choice(vratio);




		//vector <float> vratio;

		vratio = flt::fvector::arange(ratio_min.h + 0.05, ratio_max.h - 0.05, 0.1);

		if(!vratio.size())

			ratio.h = 0.5;

		else

			ratio.h = random_choice(vratio);

		//ratio.h = random_choice(tool::xvector::arange(ratio_min.h + 0.05, ratio_max.h - 0.05, 0.1));








		data::cBox cnproposal = mapping::slide(ratio, sproposal, cbox);

		//cout << "sproposal w : " << sproposal.width << endl;

		//cout << "sproposal h : " << sproposal.height << endl;

		data::bBox bnproposal = mapping::center_to_border(cnproposal, sin);

		//mapping::draw(in, bbox, "bnproposal");

		data::ulBox ulnproposal = mapping::center_to_upper_left(cnproposal);

		data::bBox bnbox = mapping::mapping_object(bnproposal, bbox);

		//cout << "bnproposal xmin: " << bnproposal.xmin << endl;
		//cout << "bnproposal ymin: " << bnproposal.ymin << endl;
		//cout << "bnproposal xmax: " << bnproposal.xmax << endl;
		//cout << "bnproposal ymax: " << bnproposal.ymax << endl;


		cv::Mat nproposal = mapping::crop(in, bnproposal);


		//mapping::draw(&nproposal, bnbox, "nproposal");

		cout << "cropping" << endl;

		cv::Size snproposal;

		snproposal.width = nproposal.size().width;

		snproposal.height = nproposal.size().height;


		/* scale cropped proposal to original training size */

		bnbox = mapping::resize_to_train_size(bnbox, snproposal, strain);

		//cv::Size _strain;



		cv::resize(nproposal, (*in), strain);

		data::ulBox ulnbox = mapping::border_to_upper_left(bnbox);

		return ulnbox;
	}



	data::ulBox augmentation(cv::Mat *in, data::ulBox ulbox, augmentation::method method){

		cout << "============== augmentation ===============" << endl;
		//unsigned seed = (unsigned)time(NULL); // 取得時間序列

		//srand(getpid());


		/*

		parameters setting

		*/


		vector <float> saturation_ratio {0.1, 0.2, 0.3, 0.4, 0.5};

		vector <float> hue_ratio {0.1, 0.2, 0.3, 0.4, 0.5};

		vector <float> relsolution_ratio {0.1, 0.2, 0.3, 0.4, 0.5};

		vector <float> brightness_ratio {0.1, 0.2, 0.3, 0.4, 0.5};


		cout << "in w : " << (*in).size().width << endl;

		cout << "in h : " << (*in).size().height << endl;


		cv::Mat hsv;


		/*

		RGB color  augmentation

		*/

		ulbox = flip(in, ulbox);

		data::bBox bbox = mapping::upper_left_to_border(ulbox);

		//mapping::draw(in, bbox, "flip");

		//relsolution(in, relsolution_ratio);


		/*

		HSV color space augmentation

		*/

		cv::cvtColor((*in), hsv, CV_BGR2HSV); // convert to HSV color space

		saturation(&hsv, saturation_ratio);

		hue(&hsv, hue_ratio);

		brightness(&hsv, brightness_ratio);

		cv::cvtColor(hsv, (*in), CV_HSV2BGR); // convert back to RGB color space


		cv::Size strain;

		strain.width = (*in).size().width;

		strain.height = (*in).size().height;

		ulbox = random_proposal(in, ulbox, strain);

		return ulbox;
	}
}

float random_choice(vector <float> v){

	/*

	return vector item randomly

	*/

	//cout << "v size : " << v.size() << endl;


	return v[rand() % v.size()];
}

/*

int main(){

	cv::Mat image = cv::imread("test.jpg", cv::IMREAD_COLOR);


	//cv::Mat cropped = augmentation::crop(&image, 50, 50, 100, 100);


	augmentation::method method;

	// load configuration

	augmentation::augmentation(& image, method);

	cv::imshow("Display window", image);

	cv::waitKey(0);

}

*/


#endif
