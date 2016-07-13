#ifndef CAFFE_UTIL_CUDNN_H_
#define CAFFE_UTIL_CUDNN_H_
#ifdef USE_CUDNN

#include <cudnn.h>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

#define CUDNN_VERSION_MIN(major, minor, patch) \
    (CUDNN_VERSION >= (major * 1000 + minor * 100 + patch))

#define CUDNN_CHECK(condition) \
  do { \
    cudnnStatus_t status = condition; \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS) << " "\
      << cudnnGetErrorString(status); \
  } while (0)

inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cudnn status";
}

namespace caffe {

namespace cudnn {

template <typename Dtype> class dataType;
template<> class dataType<float>  {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_FLOAT;
  static float oneval, zeroval;
  static const void *one, *zero;
};
template<> class dataType<double> {
 public:
  static const cudnnDataType_t type = CUDNN_DATA_DOUBLE;
  static double oneval, zeroval;
  static const void *one, *zero;
};

template <typename Dtype>
inline void createTensor4dDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void createTensorNdDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w,
    int stride_n, int stride_c, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, dataType<Dtype>::type,
        n, c, h, w, stride_n, stride_c, stride_h, stride_w));
}

template <typename Dtype>
inline void setTensor4dDesc(cudnnTensorDescriptor_t* desc,
    int n, int c, int h, int w) {
  const int stride_w = 1;
  const int stride_h = w * stride_w;
  const int stride_c = h * stride_h;
  const int stride_n = c * stride_c;
  setTensor4dDesc<Dtype>(desc, n, c, h, w,
                         stride_n, stride_c, stride_h, stride_w);
}

template <typename Dtype>
inline void setTensorNdDesc(cudnnTensorDescriptor_t* desc,
    const int total_dims,
    const int* shape, const int* stride) {

  // Pad to at least 4 dimensions
  int cudnn_dims = std::max(total_dims, 4);
  int padding = std::max(0, cudnn_dims - total_dims);

  std::vector<int> shape_int(cudnn_dims);
  std::vector<int> stride_int(cudnn_dims);

  for (int i = cudnn_dims - 1; i >= 0; --i) {
    if (i < padding) {
      shape_int[i] = 1;
      stride_int[i] = shape_int[i + 1] * stride_int[i + 1];
    } else {
      shape_int[i] = shape[i - padding];
      stride_int[i] = stride[i - padding];
    }
  }

  const int* shape_ptr = &shape_int[0];
  const int* stride_ptr = &stride_int[0];

  CUDNN_CHECK(
      cudnnSetTensorNdDescriptor(*desc, dataType<Dtype>::type, cudnn_dims,
                                 shape_ptr, stride_ptr));
}

template <typename Dtype>
inline void setTensorNdDesc(cudnnTensorDescriptor_t* desc,
    const int total_dims, const int* shape) {

  std::vector<int> full_shape(total_dims);
  std::vector<int> full_stride(total_dims);

  for (int i = total_dims - 1; i >= 0; --i) {
    full_shape[i] = shape[i];
    if (i == total_dims - 1) {
      full_stride[i] = 1;
    } else {
      full_stride[i] = full_stride[i + 1] * full_shape[i + 1];
    }
  }

  setTensorNdDesc<Dtype>(desc, total_dims,
                         &full_shape[0],
                         &full_stride[0]);
}

template <typename Dtype>
inline void setTensorNdDesc(cudnnTensorDescriptor_t* desc,
    const int num_spatial_dims,
    const int n, const int c, const int* shape) {

  std::vector<int> full_shape(num_spatial_dims + 2);
  std::vector<int> full_stride(num_spatial_dims + 2);

  full_shape[0] = n;
  full_shape[1] = c;

  for (int i = num_spatial_dims + 1; i >= 0; --i) {
    full_shape[i] = i > 1 ? shape[i-2] : full_shape[i];
    if (i == num_spatial_dims + 1) {
      full_stride[i] = 1;
    } else {
      full_stride[i] = full_stride[i + 1] * full_shape[i + 1];
    }
  }

  setTensorNdDesc<Dtype>(desc, num_spatial_dims + 2,
                         &full_shape[0],
                         &full_stride[0]);
}

template <typename Dtype>
inline void createFilterDesc(cudnnFilterDescriptor_t* desc,
    int n, int c, int h, int w) {
  CUDNN_CHECK(cudnnCreateFilterDescriptor(desc));
#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetFilter4dDescriptor(*desc, dataType<Dtype>::type,
      CUDNN_TENSOR_NCHW, n, c, h, w));
#else
  CUDNN_CHECK(cudnnSetFilter4dDescriptor_v4(*desc, dataType<Dtype>::type,
      CUDNN_TENSOR_NCHW, n, c, h, w));
#endif
}

template <typename Dtype>
inline void createConvolutionDesc(cudnnConvolutionDescriptor_t* conv) {
  CUDNN_CHECK(cudnnCreateConvolutionDescriptor(conv));
}

template <typename Dtype>
inline void setConvolutionDesc(cudnnConvolutionDescriptor_t* conv,
    cudnnTensorDescriptor_t bottom, cudnnFilterDescriptor_t filter,
    int pad_h, int pad_w, int stride_h, int stride_w) {
  CUDNN_CHECK(cudnnSetConvolution2dDescriptor(*conv,
      pad_h, pad_w, stride_h, stride_w, 1, 1, CUDNN_CROSS_CORRELATION));
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    const int num_spatial_dims,
    const int* shape,
    const int* pad, const int* stride) {
  switch (poolmethod) {
  case PoolingParameter_PoolMethod_MAX:
    *mode = CUDNN_POOLING_MAX;
    break;
  case PoolingParameter_PoolMethod_AVE:
    *mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDNN_CHECK(cudnnCreatePoolingDescriptor(pool_desc));

  std::vector<int> shape_int(num_spatial_dims);
  std::vector<int> pad_int(num_spatial_dims);
  std::vector<int> stride_int(num_spatial_dims);

  for (int i = 0; i < num_spatial_dims; ++i) {
    shape_int[i] = shape[i];
    pad_int[i] = pad[i];
    stride_int[i] = stride[i];
  }

  const int* shape_ptr = &shape_int[0];
  const int* pad_ptr = &pad_int[0];
  const int* stride_ptr = &stride_int[0];

#if CUDNN_VERSION_MIN(5, 0, 0)
  CUDNN_CHECK(cudnnSetPoolingNdDescriptor(*pool_desc,
                                          *mode,
                                          CUDNN_PROPAGATE_NAN,
                                          num_spatial_dims,
                                          shape_ptr,
                                          pad_ptr,
                                          stride_ptr));
#else
  CUDNN_CHECK(cudnnSetPoolingNdDescriptor_v4(*pool_desc,
                                          *mode,
                                          CUDNN_PROPAGATE_NAN,
                                          num_spatial_dims,
                                          shape_ptr,
                                          pad_ptr,
                                          stride_ptr));
#endif
}

template <typename Dtype>
inline void createPoolingDesc(cudnnPoolingDescriptor_t* pool_desc,
    PoolingParameter_PoolMethod poolmethod, cudnnPoolingMode_t* mode,
    int h, int w, int pad_h, int pad_w, int stride_h, int stride_w) {

  const int num_spatial_dims = 2;
  const int shape[2] = {h, w};
  const int pad[2] = {pad_h, pad_w};
  const int stride[2] = {stride_h, stride_w};
  createPoolingDesc<Dtype>(pool_desc,
                           poolmethod,
                           mode,
                           num_spatial_dims,
                           &shape[0],
                           &pad[0],
                           &stride[0]);
}

template <typename Dtype>
inline void createActivationDescriptor(cudnnActivationDescriptor_t* activ_desc,
    cudnnActivationMode_t mode) {
  CUDNN_CHECK(cudnnCreateActivationDescriptor(activ_desc));
  CUDNN_CHECK(cudnnSetActivationDescriptor(*activ_desc, mode,
                                           CUDNN_PROPAGATE_NAN, Dtype(0)));
}

}  // namespace cudnn

}  // namespace caffe

#endif  // USE_CUDNN
#endif  // CAFFE_UTIL_CUDNN_H_
