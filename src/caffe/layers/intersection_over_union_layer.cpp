#include <algorithm>
#include <vector>

#include "caffe/layers/intersection_over_union_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe{
template <typename Dtype>
void IntersectionOverUnionLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
	CHECK_EQ(bottom[0]->count()/bottom[0]->shape(2), bottom[1]->count())
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
	const Dtype* top_data = top[0]->mutable_cpu_data();
	caffe_set(top[0]->count(),Dtype(0),top_data);
	const int num=bottom[0]->shape(0);
	const int classes=bottom[0]->shape(1);
	const int height=bottom[0]->shape(2);
	const int width=bottom[0]->shape(3);
	

	double IUscore = 0.0;
	// n_ii  number of correctly classified pixels
	int True_positive = 0;
	// n_ij  number of pixels of class i predicted to j 
	//int False_positive = 0;
	// n_ji number of pixels of class j predicted to i
	int False_negative = 0;
	// ti  total number of pixels in class i
	int T_i =0;

	for(int i = 0; i < num; ++i){
		True_positive =0;
		False_positive=0;
		Ti = 0;
		IUscore=0;
		for(int c = 0; c < classes; ++c){	
			True_positive = 0;
			
			// calculate number of correctly classified pixels in class i
			for(int h = 0; h < height; ++h){
				for(int w = 0; w < width; ++w){
					const int idx = (c * height + h) * width + w;
					const int label_idx = h * width + w;
					if (bottom_data[idx]==bottom_label[label_idx])
						True_positive++;
				}
			}
			// sum of n_ji: total number of pixels of class j predicted to i
			for(int class_idx=0;class_idx < classes; ++class_idx){
				for(int pixel_idx=0; pixel_idx < height * width; ++pixel_idx){
					if (bottom_data[class_idx*classes+pixel_idx] == c)
						T_i++;
					if (bottom_label[pixel_idx]==c)
						False_negative++;
				}
			}
			
			//calculate IU for each class
			IUscore += True_positive /(T_i + False_negative - True_positive);
		}

		top_data[i] = IUscore / classes;
		bottom_data += bottom[0]->offset(0,1);
		bottom_label += bottom[1]->offset(0,1);
		//increment when complete computing each image
	}

}


INSTANTIATE_CLASS(IntersectionOverUnionLayer);
REGISTER_LAYER_CLASS(IntersectionOverUnion);


} //namespace caffe