#ifndef CAFFE_INTERSECTION_OVER_UNION_LAYER_HPP_
#define CAFFE_INTERSECTION_OVER_UNION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {
/*
This function takes the prediction and label of a single image, returns intersection and union areas for each class
 To compute over many images do:
 for i = 1:Nimages
  [area_intersection(:,i), area_union(:,i)]=intersectionAndUnion(imPred{i}, imLab{i});
 end
 IoU = sum(area_intersection,2)./sum(eps+area_union,2);
*/

template <typename Dtype>
class IntersectionOverUnionLayer : public Layer<Dtype> {
 public:
  explicit IntersectionOverUnionLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "IntersectionOverUnion"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    for (int i = 0; i < propagate_down.size(); ++i) {
      if (propagate_down[i]) { NOT_IMPLEMENTED; }
    }
  }

  //int label_axis_, outer_num_, inner_num_;



};


}  // namespace caffe

#endif  // CAFFE_INTERSECTION_OVER_UNION_LAYER_HPP_
