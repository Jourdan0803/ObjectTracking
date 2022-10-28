from absl import app
import numpy as np
from yolo_model.models import Yolo


YOLOV3_LAYER_LIST = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]


def load_darknet_weights(model, weights_file):
    file = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(file, dtype=np.int32, count=5)
    layers = YOLOV3_LAYER_LIST
    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
                    sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]
            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(file, dtype=np.float32, count=filters)
            else:
                bn_weights = np.fromfile(file, dtype=np.float32, count=4 * filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            # darknet shape (out_dim, in_dim, height, width)
            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                file, dtype=np.float32, count=np.product(conv_shape))
            # tf shape (height, width, in_dim, out_dim)
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(file.read()) == 0, 'failed to read all data'
    file.close()


def main(_argv):
    yolo = Yolo(classes=80)
    yolo.summary()
    load_darknet_weights(yolo, 'data/weight/yolov3.weights')
    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    yolo.save_weights('data/weight/yolov3.tf')
    print('load finished\n')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
