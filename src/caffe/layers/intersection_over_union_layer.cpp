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
					if(bottom_data[pred_idx]== 1 && bottom_label[label_idx] == class_idx+1) 
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
						if (bottom_data[pred_idx]== 1 && bottom_label[label_idx]== class_idx+1)
							G_i++;
					}
				}
				//std::cout<<std::endl;
			}
		}
		//calculate P_i
		//predicting class_idx, ground truth in all class
		for(int n = 0; n < num;n++){
			for(int i = 0; i < classes;i++){
				for(int h = 0; h < height; h++){
					for(int w = 0; w < width; w++){
						const int pred_idx = ((n * classes + class_idx) * height + h) * width + w;
						const int label_idx = (n * height + h) * width + w;
							//std::cout << "pred_idx: "<< pred_idx << std::endl;
							//std::cout << "bottom_data: " << bottom_data[pred_idx]<< std::endl;
							//std::cout << "bottom_label: "<< bottom_label[label_idx] << std::endl;
						if(bottom_data[pred_idx]==1 && bottom_label[label_idx] == i+1){
							P_i++;
							//std::cout << "TRUE" << std::endl;
						}
					}
				}
			}
		}
		//calculate IU for each class
		//std::cout << "C_i:"<< C_i <<std::endl;
		//std::cout << "G_i: "<< G_i <<std::endl;
		//std::cout << "P_i: "<< P_i <<"\n"<<std::endl;
		
		IUscore +=  (float)C_i/(G_i + P_i - C_i);
	}
	
	top_data[0] = IUscore / classes;
	//std::cout<<top_data[0]<<std::endl;
}



template <typename Dtype>
void IntersectionOverUnionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom){

        const Dtype* bottom_data = bottom[0]->mutable_cpu_data();
        
        const Dtype* bottom_label= bottom[1]->mutable_cpu_data();
        
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        

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

        
        // C_i  number of correctly classified pixels in class i. 
        double C_i = 0;
        
        // G_i  total number of pixels whose label is i
        double G_i = 0;
        
        // P_i: total number of pixels whose prediction is i
        double P_i = 0;
        

        for (int class_idx = 0; class_idx < classes; class_idx++){
        		//label i, predict i
                C_i=0;
                vector<int> ii;
				//label i, predict j
                G_i=0;
                vector<int> ij;
                //label j, predict i
                P_i=0;
                vector<int> ji;
                
                for(int idx = 0; idx < num; idx++){
                        
                                
                                        //const int label_idx = (n * height + h) * width + w;
                                        
                                        
                                        if( bottom_data[idx] == bottom_label[idx] ) {
											C_i++;
											ii.push_back(idx);
                                        }

                                        if( bottom_label[idx] == class_idx ) {
                                        	G_i++;	
                                        	ij.push_back(idx);
                                        }

                                        if( bottom_data[idx] == class_idx ) {
                                        	P_i++;	
                                        	ji.push_back(idx);
                                        }
                        
                        
                }
                
                std::cout<<"class :"<< class_idx <<std::endl;
				std::cout<<"C_i "<< C_i<<std::endl;
				std::cout<<"G_i "<< G_i<<std::endl;
				std::cout<<"P_i "<< P_i<<std::endl;

				double gradient_C_i_constant = (double)((G_i + P_i - 2*C_i));
                double gradient_C_i =  gradient_C_i_constant/((double)(pow(gradient_C_i_constant+C_i,2)));
                double gradient_G_i_P_i = (-1)*C_i/((double)(pow(G_i + P_i - C_i,2)));

                std::cout<<"gradient_C_i_constant "<< gradient_C_i_constant<<std::endl;
                std::cout<<"gradient_C_i "<< gradient_C_i<<std::endl;
                std::cout<<"gradient_gradient_G_i_P_iC_i_constant "<<gradient_G_i_P_i <<std::endl;

                for(int i = 0;i < ii.size();i++){
                        int idx = ii[i];
                        bottom_diff[idx] = gradient_C_i;
                }

                for(int i = 0;i < ij.size();i++){
                        int idx = ij[i];
                        bottom_diff[idx] = gradient_G_i_P_i;
                }

                for(int i = 0;i < ji.size();i++){
                        int idx = ji[i];
                        bottom_diff[idx] = gradient_G_i_P_i;
                }
        }

        std::cout<<"done"<<std::endl;
        //test
        std::cout<<bottom_diff[0]<<std::endl;
        std::cout<<bottom_diff[1]<<std::endl;
        std::cout<<bottom_diff[2]<<std::endl;
        std::cout<<bottom_diff[4]<<std::endl;
}



INSTANTIATE_CLASS(IntersectionOverUnionLayer);
REGISTER_LAYER_CLASS(IntersectionOverUnion);
} //namespace caffe
                                                        