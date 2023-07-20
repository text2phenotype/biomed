import tensorflow as tf

# Enforce eager execution for all integration tests
# Without this, TF1 models will try to use Graph/Session and raise compatibility errors
tf.compat.v1.enable_eager_execution(
    config=None, device_policy=None, execution_mode=None
)
