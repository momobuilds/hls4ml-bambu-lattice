import hls4ml
import tensorflow as tf

# 1. Load the brain
model = tf.keras.models.load_model('models/tiny_mnist_model.h5')

# 2. Create the hardware configuration
# We use 'Bambu' as the backend for Lattice support
config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Bambu')

# 3. Tuning for Lattice (ECP5 is small, so we optimize for space)
config['Model']['Precision'] = 'ap_fixed<16,6>' 
config['Model']['ReuseFactor'] = 4 

# 4. Convert to C++
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls_output',
    backend='Bambu'
)

# 5. Check if it compiles locally (software simulation)
hls_model.compile()
print("C++ code is ready in /hls_output!")