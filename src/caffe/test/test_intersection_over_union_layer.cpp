#include <cfloat>
#include <vector>
#include <iostream>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/intersection_over_union_layer.hpp"
#include "caffe/util/rng.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {
template <typename Dtype>
class IntersectionOverUnionLayerTest : public CPUDeviceTest<Dtype>{
protected:
	IntersectionOverUnionLayerTest()
	:	blob_bottom_data_(new Blob<Dtype>(2,3,2,2)),
		blob_bottom_label_(new Blob<Dtype>(2,1,2,2)),
		blob_top_(new Blob<Dtype>(1,1,1,1)){
		//fill values	
		FillerParameter filler_param;
	    GaussianFiller<Dtype> filler(filler_param);
	    filler.Fill(this->blob_bottom_data_);
	    blob_bottom_vec_.push_back(blob_bottom_data_);
	    filler.Fill(this->blob_bottom_label_);
	    blob_bottom_vec_.push_back(blob_bottom_label_);
	    blob_top_vec_.push_back(blob_top_);
	}
	~IntersectionOverUnionLayerTest(){
		delete blob_bottom_data_;
		delete blob_bottom_label_;
		delete blob_top_;
	}
	Blob<Dtype>* const blob_bottom_data_;
  	Blob<Dtype>* const blob_bottom_label_;
  	Blob<Dtype>* const blob_top_;
  	vector<Blob<Dtype>*> blob_bottom_vec_;
  	vector<Blob<Dtype>*> blob_top_vec_;

};
	

TYPED_TEST_CASE(IntersectionOverUnionLayerTest, TestDtypes);

TYPED_TEST(IntersectionOverUnionLayerTest, TestSetup) {
  LayerParameter layer_param;
  IntersectionOverUnionLayer<TypeParam> layer(layer_param);
  
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  // check bottom[0]
  EXPECT_EQ(this->blob_bottom_data_->num(),2);
  EXPECT_EQ(this->blob_bottom_data_->channels(),3);  
  EXPECT_EQ(this->blob_bottom_data_->height(),2);
  EXPECT_EQ(this->blob_bottom_data_->width(),2);
  //check bottom[1]
  EXPECT_EQ(this->blob_bottom_label_->num(),2);
  EXPECT_EQ(this->blob_bottom_label_->channels(),1);
  EXPECT_EQ(this->blob_bottom_label_->height(),2);
  EXPECT_EQ(this0>blob_bottom_label_->width(),2);

  //check top
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(IntersectionOverUnionLayerTest, TestForward){
	LayerParameter layer_param;
	IntersectionOverUnionLayer<TypeParam> layer(layer_param);
	/* Input : bottom_data 2*3*2*2
		image 0: 
		class0	[1         -FLT_MAX]
				[-FLT_MAX  -FLT_MAX]

		class1	[-FLT_MAX 	      1]
				[1		   -FLT_MAX]

		class2	[-FLT_MAX  -FLT_MAX]
				[-FLT_MAX         1]
		
		image 1:
		class0	[-FLT_MAX         1]
				[-FLT_MAX         1]
		
		class1 	[-FLT_MAX  -FLT_MAX]
				[1         -FLT_MAX]

		class2  [1         -FLT_MAX]
				[-FLT_MAX  -FLT_MAX]

		Input : bottom_label 2*1*2*2
	   	image 0: [1 	0]
	   			 [2 	2]
	   	image 1: [2 	0]
				 [1 	2]
		
	*/
	//prediction
	//image 0
	//class 0
	this->blob_bottom_data_->mutable_cpu_data()[0] = 1;
	this->blob_bottom_data_->mutable_cpu_data()[1] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[2] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[3] = -FLT_MAX;
	//class 1
	this->blob_bottom_data_->mutable_cpu_data()[4] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[5] = 1;
	this->blob_bottom_data_->mutable_cpu_data()[6] = 1;
	this->blob_bottom_data_->mutable_cpu_data()[7] = -FLT_MAX;
	//class 2
	this->blob_bottom_data_->mutable_cpu_data()[8] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[9] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[10] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[11] = 1;

	//image 1
	//class 0
	this->blob_bottom_data_->mutable_cpu_data()[12] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[13] = 1;
	this->blob_bottom_data_->mutable_cpu_data()[14] = 1;
	this->blob_bottom_data_->mutable_cpu_data()[15] = -FLT_MAX;
	//class 1
	this->blob_bottom_data_->mutable_cpu_data()[16] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[17] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[18] = 1;
	this->blob_bottom_data_->mutable_cpu_data()[19] = -FLT_MAX;
	//class 2
	this->blob_bottom_data_->mutable_cpu_data()[20] = 1;
	this->blob_bottom_data_->mutable_cpu_data()[21] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[22] = -FLT_MAX;
	this->blob_bottom_data_->mutable_cpu_data()[23] = -FLT_MAX;
	
	//label
	//image 0
	this->blob_bottom_label_->mutable_cpu_data()[0]= 1;
	this->blob_bottom_label_->mutable_cpu_data()[1]= 0;
	this->blob_bottom_label_->mutable_cpu_data()[2]= 2;
	this->blob_bottom_label_->mutable_cpu_data()[3]= 2;
	//image 1
	this->blob_bottom_label_->mutable_cpu_data()[4]= 2;
	this->blob_bottom_label_->mutable_cpu_data()[5]= 0;
	this->blob_bottom_label_->mutable_cpu_data()[6]= 1;
	this->blob_bottom_label_->mutable_cpu_data()[7]= 2;
    

	
	//Forward test
	layer.Forward(this->blob_bottom_vec_,this->blob_top_vec_);
	
	//std::cout<<

	/* Expected output: 
	
	*/
	//EXPECT_NEAR(this->blob_top_->cpu_data()[0], 0, 1e-4);
}


} //namespace caffe