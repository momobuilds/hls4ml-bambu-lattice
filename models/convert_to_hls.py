import hls4ml
import tensorflow as tf

# 1. Load your trained model
model = tf.keras.models.load_model('tiny_mnist_model.h5')

# 2. Create a basic configuration
# 'granularity=name' allows you to customize specific layers later if needed
config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Bambu')

# 3. Set FPGA-specific optimizations
# Lattice ECP5 is small, so we'll use a ReuseFactor to save space
config['Model']['ReuseFactor'] = 4 
config['Model']['Precision'] = 'ap_fixed<16,6>' # 16 bits total, 6 for integer

# 4. Perform the conversion
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls_output', # This folder will be created
    backend='Bambu',
    part='Lattice'           # Explicitly tell it we are not using Xilinx
)

# 5. Compile and generate the C++ project
hls_model.compile()
print("Success! C++ code generated in the 'hls_output' folder.")