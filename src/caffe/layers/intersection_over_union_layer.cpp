#include <utility>
#include <vector>
#include <functional>
#include <cfloat>
#include <iostream>
#include <cstdio>

#include "caffe/layers/intersection_over_union_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe{
template <typename Dtype>
void IntersectionOverUnionLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	CHECK_EQ(bottom[0]->count()/bottom[0]->shape(1), bottom[1]->count())
		<< "Number of labels must match number of predictions; "
      	<< "e.g., if prediction shape is (N, C, H, W), "
      	<< "label count (number of labels) must be N*H*W."
      	<< "bottom[0] N*C*H*W: " << bottom[0]->count()
      	<< "bottom[0] C: " << bottom[0]->count()
      	<< "bottom[1] N*1*H*W: " << bottom[1]->count();
}


template <typename Dtype>
void IntersectionOverUnionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
	const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
	const Dtype* bottom_label= bottom[1]->mutable_cpu_data();
	Dtype* top_data = top[0]->mutable_cpu_data();
	const int num = bottom[0]->num();
	const int classes = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	/*
	std::cout<< "Num: "<< bottom[0]->num() << std::endl;
	std::cout<< "Classes:"<< bottom[0]->channels() << std::endl;
	std::cout<< "Height:"<< bottom[0]->height() << std::endl;
	std::cout<< "Width:"<< bottom[0]->width() << std::endl;
	*/

	float IUscore = 0.0;
	// C_i  number of correctly classified pixels in class i. 
	int C_i = 0;
	// G_i  total number of pixels whose label is i
	int G_i = 0;
	// P_i: total number of pixels whose prediction is i
	int P_i = 0;


	for (int class_idx = 0; class_idx < classes; class_idx++){
		C_i=0;
		G_i=0;
		P_i=0;
		//calculate C_i
		for(int n = 0; n < num; n++){
			for(int h = 0; h < height; h++){
				for(int w = 0; w < width; w++){
					const int pred_idx = ((n * classes + class_idx) * height + h) * width + w;
					const int label_idx = (n * height + h) * width + w;
					//std::cout << "pred_idx: "<< pred_idx << std::endl;
					//std::cout << "bottom_data: " << bottom_data[pred_idx]<< std::endl;
					if(bottom_data[pred_idx]== 1 && bottom_label[label_idx] == class_idx) 
						C_i++;
				}
			}
		}
		//calculate G_i. 
		// prediction in all class,  ground truth in class_idx
		for(int n = 0; n < num;n++){
			for(int i = 0 ; i < classes; i++){
				for(int h = 0; h < height; h++){
					for(int w = 0; w < width; w++){
						const int pred_idx = ((n * classes + i) * height + h) * width + w;
						const int label_idx = (n * height + h) * width + w;
						if (bottom_data[pred_idx]== 1 && bottom_label[label_idx]== class_idx)
							G_i++;
					}
				}
			}
		}
		//calculate P_i
		//predicting class_idx, ground truth in all class
		for(int n = 0; n < num;n++){
			for(int i =0; i< classes;i++){
				for(int h = 0; h < height; h++){
					for(int w = 0; w < width; w++){
						const int pred_idx = ((n * classes + i) * height + h) * width + w;
						const int label_idx = (n * height + h) * width + w;
						std::cout << "pred_idx: "<< pred_idx << std::endl;
							std::cout << "bottom_data: " << bottom_data[pred_idx]<< std::endl;
						if(bottom_label[label_idx] == class_idx){
							if(bottom_data[pred_idx]==1){
								P_i++;
								std::cout << "TRUE" << std::endl;
							}
						}
					}
				}
				std::cout<<std::endl;
			}
		}
		//calculate IU for each class
		std::cout << "C_i:"<< C_i <<std::endl;
		std::cout << "G_i: "<< G_i <<std::endl;
		std::cout << "P_i: "<< P_i <<"\n"<<std::endl;
		
		IUscore += C_i /(G_i + P_i - C_i);
	}
	std::cout << "IUscore : " << IUscore << std::endl;
	top_data[0] = IUscore / classes;
}
		

		


INSTANTIATE_CLASS(IntersectionOverUnionLayer);
REGISTER_LAYER_CLASS(IntersectionOverUnion);

} //namespace caffe