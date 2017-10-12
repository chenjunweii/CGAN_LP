#ifndef AUGMENTATION_H
#define AUGMENTATION_H
#include <iostream>
#include "data.h"
//#include <tool.hh>
#include <opencv2/opencv.hpp>











namespace augmentation{

	enum HSV
	{
		H = 0,
		S = 1,
		V = 2,
	};


	struct method
	{
		bool flip;
		bool saturation;
		bool hue;
		bool random_proposal;
		bool gaussian_blur;	
	};

	data::ulBox flip(cv::Mat *in);

	int relsolution(cv::Mat *in);

	int brightness(cv::Mat *in);

	int saturation(cv::Mat *in, vector <float> v);

	int hue(cv::Mat *in, vector <float> v);

	int gaussian_blur(cv::Mat *in);

	data::ulBox random_proposal(cv::Mat *in, data::ulBox box, data::SIZE strain);

	data::ulBox augmentation(cv::Mat *in, data::ulBox box, method method);


	namespace mapping{

		data::cBox upper_left_to_center(data::ulBox);

		data::ulBox center_to_upper_left(data::cBox);

		data::bBox upper_left_to_border(data::ulBox);

		data::bBox center_to_border(data::cBox);

		data::bBox center_to_border(data::cBox, cv::Size);

		data::ulBox border_to_upper_left(data::bBox);

		data::cBox border_to_center(data::bBox);

		data::cBox slide(data::RATIO ratio, cv::Size scaled, data::cBox box);

		data::bBox mapping_object(data::bBox proposal, data::bBox box);

		data::bBox resize_to_train_size(data::bBox, cv::Size, cv::Size);

		//cv::Mat crop(cv::Mat, bBox proposal);

		cv::Mat crop(cv::Mat *in, data::bBox box);

		int draw(cv::Mat *in, data::bBox box, char *);
		
	}

}

float random_choice(vector <float> v);



#endif
