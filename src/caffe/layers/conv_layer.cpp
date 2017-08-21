#include <vector>

#include "caffe/layers/conv_layer.hpp"

#include "perfdnn.h"


namespace caffe {


template <typename Dtype>
enum perfdnn_status caffe_perfdnn_convolution_inference(
        enum perfdnn_convolution_algorithm algorithm,
        size_t input_channels,
        size_t output_channels,
        struct perfdnn_size input_size,
        struct perfdnn_padding input_padding,
        struct perfdnn_size kernel_size,
        struct perfdnn_size output_subsampling,
        struct perfdnn_size dilation,
        const Dtype* input,
        const Dtype* kernel,
        Dtype* bias,
        Dtype* output,
        pthreadpool_t threadpool,size_t group);


template <>
enum perfdnn_status caffe_perfdnn_convolution_inference<double>(
        enum perfdnn_convolution_algorithm algorithm,
        size_t input_channels,
        size_t output_channels,
        struct perfdnn_size input_size,
        struct perfdnn_padding input_padding,
        struct perfdnn_size kernel_size,
        struct perfdnn_size output_subsampling,
        struct perfdnn_size dilation,
        const double* input,
        const double* kernel,
        double* bias,
        double* output,
        pthreadpool_t threadpool, size_t group){
  return perfdnn_status_unsupported_algorithm;
}



template <>
enum perfdnn_status caffe_perfdnn_convolution_inference<float>(
        enum perfdnn_convolution_algorithm algorithm,
        size_t input_channels,
        size_t output_channels,
        struct perfdnn_size input_size,
        struct perfdnn_padding input_padding,
        struct perfdnn_size kernel_size,
        struct perfdnn_size output_subsampling,
        struct perfdnn_size dilation,
        const float* input,
        const float* kernel,
        float* bias,
        float* output,
        pthreadpool_t threadpool, size_t group){
  return perfdnn_convolution_inference(algorithm, input_channels, output_channels, input_size, input_padding, kernel_size, output_subsampling, dilation, input, kernel, bias, output, threadpool, group);
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

/*
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}
*/


//this modification is for the test PerfDNN convolution function in some special layer
static int cnt=0;
template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {


    int width = static_cast<size_t>(this->blobs_[0]->width());
    int height = static_cast<size_t>(this->blobs_[0]->height());

    const int* dilation_data = this->dilation_.cpu_data();

//if(width!=3||height!=3||this->group_!=1)
//if(this->group_!=1||width!=11)
//if(dilation_data[0]>1)
//if(!cnt)
    
    if(0){
	//cnt++;


	const Dtype* weight = this->blobs_[0]->cpu_data();
	for (int i = 0; i < bottom.size(); ++i) {

	  const Dtype* bottom_data = bottom[i]->cpu_data();
	  Dtype* top_data = top[i]->mutable_cpu_data();

	  int width = static_cast<size_t>(bottom[i]->width());
	  int height = static_cast<size_t>(bottom[i]->height());

	  struct timeval conv_start, conv_end;
	  //gettimeofday(&conv_start,NULL);

	  for (int n = 0; n < this->num_; ++n) {
	    this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
				   top_data + n * this->top_dim_);
	    if (this->bias_term_) {
	      const Dtype* bias = this->blobs_[1]->cpu_data();
	      this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
	    }
	  }

 
	  //gettimeofday(&conv_end,NULL);

	  //float conv_time_use = 1000.0*(float)(conv_end.tv_sec-conv_start.tv_sec)+(float)(conv_end.tv_usec-conv_start.tv_usec)/1000.0;
	  //printf("conv Time use = %f(ms)\n",conv_time_use);

	}


      }else{

      enum perfdnn_status init_status = perfdnn_initialize();

      const Dtype* weight = this->blobs_[0]->cpu_data();
      //CHECK(this->bias_term_);

      Dtype* bias = (Dtype*)this->blobs_[1]->cpu_data();
      //Dtype* bias = NULL;
      //std::cout << "my bias " << bias << std::endl;
  

      for (int i = 0; i < bottom.size(); ++i) {
	const Dtype* bottom_data = bottom[i]->cpu_data();
	Dtype* top_data = top[i]->mutable_cpu_data();

	const size_t batch_size = bottom[i]->num();

	const size_t input_channels = bottom[i]->channels();
	const size_t output_channels = top[i]->channels();

	perfdnn_size input_size;
	input_size.width = static_cast<size_t>(bottom[i]->width());
	input_size.height = static_cast<size_t>(bottom[i]->height());

	perfdnn_padding input_padding;
	input_padding.top = input_padding.bottom =
	  static_cast<size_t>(this->pad_.cpu_data()[0]);
	input_padding.left = input_padding.right =
	  static_cast<size_t>(this->pad_.cpu_data()[1]);


	perfdnn_size kernel_size;
	kernel_size.width = static_cast<size_t>(this->blobs_[0]->width());
	kernel_size.height = static_cast<size_t>(this->blobs_[0]->height());
    
	pthreadpool_t threadpool=NULL;



	//const perfdnn_status status = caffe_perfdnn_convolution_output<Dtype>(algorithm,
	//  batch_size, input_channels, output_channels,
	//  input_size, input_padding,
	//  kernel_size,
	//  bottom_data, weight, bias, top_data,
	//  static_cast<pthreadpool_t>(threadpool));

	enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_auto;
	//enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_wt8x8;
	//enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_im2col_gemm;
	//enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_ft8x8;
	//enum perfdnn_convolution_algorithm algorithm = perfdnn_convolution_algorithm_ft16x16;
	const struct perfdnn_size output_subsampling = {.width=this->stride_.cpu_data()[1], .height=this->stride_.cpu_data()[0]};
	const struct perfdnn_size dilation = {.width=this->dilation_.cpu_data()[1], .height=this->dilation_.cpu_data()[0]};


	//struct timeval conv_start, conv_end;
	size_t group=this->group_;

	//gettimeofday(&conv_start,NULL);
      
	for(int n=0; n<batch_size; n++){
         
	    const perfdnn_status status = caffe_perfdnn_convolution_inference(algorithm, input_channels, output_channels, input_size, input_padding, kernel_size, output_subsampling, dilation, bottom_data+n*this->bottom_dim_, weight, bias, top_data+n*this->top_dim_, NULL, group);
	    //const perfdnn_status status = caffe_perfdnn_convolution_inference(algorithm, input_channels, output_channels, input_size, input_padding, kernel_size, output_subsampling, dilation, bottom_data+n*this->bottom_dim_, weight, bias, top_data+n*this->top_dim_, NULL);
	    CHECK_EQ(perfdnn_status_success, status);

	  }

	//gettimeofday(&conv_end,NULL);
	//float conv_time_use = 1000.0*(float)(conv_end.tv_sec-conv_start.tv_sec)+(float)(conv_end.tv_usec-conv_start.tv_usec)/1000.0;
	//printf("perfdnn conv Time use = %f(ms)\n",conv_time_use);

      }

    }
}









template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  // namespace caffe
