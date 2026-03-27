import hls4ml
import tensorflow as tf

# 1. Load your trained Tiny-CNN
model = tf.keras.models.load_model('models/tiny_mnist_model.h5')

# 2. Generate a basic configuration
config = hls4ml.utils.config_from_keras_model(model, granularity='name')

# 3. Customize for your ECP5UM
# We set the precision to fixed-point (ap_fixed<16,6>) which is safer for FPGAs
config['Model']['Precision'] = 'ap_fixed<16,6>'
config['Model']['ReuseFactor'] = 1

# 4. Perform the Conversion
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls4ml_prj',
    backend='Bambu',
    part='LFE5UM-85F' # Matches your ECP5UM board
)

# 5. Build the Verilog
print("Building Verilog with Bambu...")
hls_model.build(reset=True, csim=False, synth=True)