#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<16,6> input_t;
typedef ap_fixed<37,17> conv1_accum_t;
typedef ap_fixed<37,17> conv1_result_t;
typedef ap_fixed<16,6> conv1_weight_t;
typedef ap_fixed<16,6> conv1_bias_t;
typedef ap_fixed<16,6> layer3_t;
typedef ap_fixed<18,8> conv1_relu_table_t;
typedef ap_fixed<16,6> pool1_accum_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<43,23> output_accum_t;
typedef ap_fixed<43,23> output_result_t;
typedef ap_fixed<16,6> output_weight_t;
typedef ap_fixed<16,6> output_bias_t;
typedef ap_uint<1> layer6_index;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_softmax_accum_t;
typedef ap_fixed<16,6> result_t;
typedef ap_fixed<18,8> output_softmax_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_softmax_exp_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_softmax_inv_table_t;
typedef ap_fixed<18,8,AP_RND,AP_SAT,0> output_softmax_inv_inp_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
