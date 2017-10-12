#ifndef AUGMENTATION_MAPPING_HH
#define AUGMENTATION_MAPPING_HH
#include <iostream>
#include <flt.h>
#include <opencv2/opencv.hpp>
#include "augmentation.h"
#include "data.h"


using namespace std;






namespace augmentation{

	namespace mapping{

		int draw(cv::Mat *in, data::bBox box, char * window){

			cv::Point ul(box.xmin, box.ymin);

			cv::Point br(box.xmax, box.ymax);

			cv::rectangle((*in), ul, br, cv::Scalar(0, 255, 255), 1);

			cv::imshow(window, (*in));

			cv::waitKey();

		}



		data::bBox resize_to_train_size(data::bBox bbox, cv::Size soriginal, cv::Size snew){

			/*
		
			when image scaled, remap the bounding box; 

			*/

			data::RATIO ratio;

			ratio.w = snew.width / soriginal.width;
			ratio.h = snew.height / soriginal.height;

			data::bBox mapped;

			mapped.xmin = bbox.xmin * ratio.w;
			mapped.ymin = bbox.ymin * ratio.h;
			mapped.xmax = bbox.xmax * ratio.w;
			mapped.ymax = bbox.ymax * ratio.h;

			return mapped;

		}

		data::bBox mapping_object(data::bBox proposal, data::bBox bbox){

			/*
			
			mapping bounding box to new proposal

			bbox : bounding box location of original proposal 

			proposal : new proposal cropped from original proposal

			width and height of original proposal will greater than new proposal 
			

			*/

			data::bBox mapped;

			mapped.xmin = bbox.xmin - proposal.xmin;
			mapped.ymin = bbox.ymin - proposal.ymin;
			mapped.xmax = bbox.xmax - proposal.xmin;
			mapped.ymax = bbox.ymax - proposal.ymin;

			return mapped;

		}


		data::cBox upper_left_to_center(data::ulBox in){

			data::cBox out;

			out.x = in.xmin + in.w * 0.5;
			out.y = in.ymin + in.h * 0.5; 
			out.w = in.w;
			out.h = in.h;

			return out;

		}

		data::ulBox center_to_upper_left(data::cBox in){

			data::ulBox out;

			out.xmin = max((int)(in.x - in.w * 0.5), 0);
			out.ymin = max((int)(in.y - in.h * 0.5), 0);
			out.w = in.w;
			out.h = in.h;

			return out;

		}

		data::bBox center_to_border(data::cBox in){

			data::bBox out;

			out.xmin = (int)(in.x - in.w * 0.5);
			out.ymin = (int)(in.y - in.h * 0.5);
			out.xmax = (int)(in.x + in.w * 0.5);
			out.ymax = (int)(in.y + in.h * 0.5);

			return out;

		}

		data::bBox center_to_border(data::cBox in, cv::Size size){

			data::bBox out;

			out.xmin = max((int)(in.x - in.w * 0.5), 0);
			out.ymin = max((int)(in.y - in.h * 0.5), 0);
			out.xmax = min((int)(in.x + in.w * 0.5), size.width);
			out.ymax = min((int)(in.y + in.h * 0.5), size.height);

			return out;

		}

		data::bBox upper_left_to_border(data::ulBox in){

			data::bBox out;

			out.xmin = in.xmin;
			out.ymin = in.ymin;
			out.xmax = in.xmin + in.w;
			out.ymax = in.ymin + in.h;

			return out;

		}

		data::ulBox border_to_upper_left(data::bBox in){

			data::ulBox out;

			out.xmin = in.xmin;
			out.ymin = in.ymin;
			out.w = in.xmax - in.xmin;
			out.h = in.ymax - in.ymin;

			return out;

		}

		data::cBox border_to_center(data::bBox in){

			data::cBox out;

			out.w = in.xmax - in.xmin;

			out.h = in.ymax - in.ymin;

			out.x = in.xmin + out.w * 0.5;

			out.y = in.ymin + out.h * 0.5;

			return out;

		}


		data::cBox slide(data::RATIO ratio, cv::Size scaled, data::cBox box){

			/*
			
			fixed the bounding box of the target object, and slides the scaled proposal window

			*/

			data::cBox mapped;

			mapped.x = (1 - ratio.w - 0.5) * scaled.width + box.x;

			mapped.y = (1 - ratio.h - 0.5) * scaled.height + box.y;

			mapped.w = scaled.width;

			mapped.h = scaled.height;

			return mapped;
		}


		cv::Mat crop(cv::Mat *in, data::bBox bbox){



			cv::Rect roi(bbox.xmin,
			 bbox.ymin,
			  min(bbox.xmax - bbox.xmin, (*in).size().width - 1),
			   min(bbox.ymax - bbox.ymin, (*in).size().height - 1));

			/*
			roi.x = ulbox.x;
			roi.y = ulbox.y;
			roi.width = ulbox.w;
			roi.height = ulbox.h;

			*/

			cout << "bbox xmin : " << bbox.xmin << endl;
			cout << "bbox ymin : " << bbox.ymin << endl;
			cout << "bbox xmax : " << bbox.xmax << endl;
			cout << "bbox ymax : " << bbox.ymax << endl;

			cout << "x : " << roi.x << endl;

			cout << "y : " << roi.y << endl;

			cout << "w : " << roi.width << endl;

			cout << "h : " << roi.height << endl;

			cout << "image w : " << (*in).size().width << endl;

			cout << "image h : " << (*in).size().height << endl;

			cv::Mat tt = (*in)(roi);

			cout << "test " << endl;
			return (*in)(roi);

		}

	}
}


#endif
