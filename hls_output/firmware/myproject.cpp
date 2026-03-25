#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_t input_layer[28*28*1],
    result_t layer7_out[10]
) {

    // hls-fpga-machine-learning insert IO
    //#pragma HLS ARRAY_RESHAPE variable=input_layer complete dim=0
    //#pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    #pragma HLS interface mode=valid port=input_layer,layer7_out 
    //#pragma HLS DATAFLOW

    // hls-fpga-machine-learning insert load weights
#ifndef __BAMBU__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<conv1_weight_t, 36>(w2, "w2.txt");
        nnet::load_weights_from_txt<conv1_bias_t, 4>(b2, "b2.txt");
        nnet::load_weights_from_txt<output_weight_t, 6760>(w6, "w6.txt");
        nnet::load_weights_from_txt<output_bias_t, 10>(b6, "b6.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    conv1_result_t layer2_out[26*26*4];
    //#pragma HLS ARRAY_PARTITION variable=layer2_out complete dim=0

    layer3_t layer3_out[26*26*4];
    //#pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    layer4_t layer4_out[13*13*4];
    //#pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0

    auto& layer5_out = layer4_out;
    output_result_t layer6_out[10];
    //#pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    nnet::conv_2d_cl<input_t, conv1_result_t, config2>(input_layer, layer2_out, w2, b2); // conv1

    nnet::relu<conv1_result_t, layer3_t, relu_config3>(layer2_out, layer3_out); // conv1_relu

    nnet::pooling2d_cl<layer3_t, layer4_t, config4>(layer3_out, layer4_out); // pool1

    nnet::dense<layer4_t, output_result_t, config6>(layer5_out, layer6_out, w6, b6); // output

    nnet::softmax<output_result_t, result_t, softmax_config7>(layer6_out, layer7_out); // output_softmax

}

