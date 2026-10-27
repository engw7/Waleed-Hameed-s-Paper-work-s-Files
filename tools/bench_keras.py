"""Framework-level latency of one Keras inference (fresh process, no determinism flags): python tools/bench_keras.py model.keras input.npy out.json"""
import sys, json, time
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
model = tf.keras.models.load_model(sys.argv[1]); xs = np.load(sys.argv[2]).astype('float32'); tfx = tf.constant(xs)
def lat(fn, n_warm, n):
    for _ in range(n_warm): fn()
    t = np.empty(n)
    for i in range(n):
        t0 = time.perf_counter_ns(); fn(); t[i] = time.perf_counter_ns() - t0
    return (t / 1e3).tolist()
out = {'call': lat(lambda: model(tfx, training=False).numpy(), 200, 3000), 'predict': lat(lambda: model.predict(xs, verbose=0), 20, 300)}
json.dump(out, open(sys.argv[3], 'w'))
