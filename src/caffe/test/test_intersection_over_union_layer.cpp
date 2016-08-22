#include <cfloat>
#include <vector>

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
	:	blob_bottom_data_(new Blob<Dtype>(1,3,2,2)),
		blob_bottom_label_(new Blob<Dtype>(1,1,2,2)),
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
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
}

TYPED_TEST(IntersectionOverUnionLayerTest, TestForward){
	LayerParameter layer_param;
	IntersectionOverUnionLayer<TypeParam> layer(layer_param);
	/* Input : bottom_data 1*3*2*2
		class 0: [0 1]
				 [1 2]
		class 1: [2 0]
				 [1 0]
		class 2: [2 1]
				 [2 2]		
	   Input : bottom_label 1*1*2*2
	   			[1 0]
	   			[2 2]
	*/
	//class 0
	blob_bottom_data_->mutable_cpu_data()[0] = 0;
	blob_bottom_data_->mutable_cpu_data()[1] = 1;
	blob_bottom_data_->mutable_cpu_data()[2] = 1;
	blob_bottom_data_->mutable_cpu_data()[3] = 2;
	//class 1
	blob_bottom_data_->mutable_cpu_data()[4] = 2;
	blob_bottom_data_->mutable_cpu_data()[5] = 0;
	blob_bottom_data_->mutable_cpu_data()[6] = 1;
	blob_bottom_data_->mutable_cpu_data()[7] = 0;
	//class 2
	blob_bottom_data_->mutable_cpu_data()[8] = 2;
	blob_bottom_data_->mutable_cpu_data()[9] = 1;
	blob_bottom_data_->mutable_cpu_data()[10] = 2;
	blob_bottom_data_->mutable_cpu_data()[11] = 2;
	
	//label
	blob_bottom_label_->mutable_cpu_data()[0]= 1;
	blob_bottom_label_->mutable_cpu_data()[1]= 0;
	blob_bottom_label_->mutable_cpu_data()[2]= 2;
	blob_bottom_label_->mutable_cpu_data()[3]= 2;


	
	//test reshape
	layer.Reshape(blob_bottom_vec_,blob_top_vec_);

	//Forward test
	layer.Forward(blob_bottom_vec_,blob_top_vec_);
	
	/* Expected output: 
		 1     1     2
	   (--- + --- + ---)/ 3 = 0.32777
		4-1   5-1   7-2
	*/
	EXPECT_NEAR(blob_top_->cpu_data()[0], 0.327, 1e-4);
}


} //namespace caffe