import os
import sys
import numpy as np
import tensorflow as tf
import utils.gif
from utils.image_processing import preprocess_file, split_image
from model import build_model



def train(imgs,
          learning_rate=0.0001,
          batch_size=200,
          n_iterations=100,
          gif_step=2,
          n_neurons=200,
          n_layers=10,
          activation_fn=tf.nn.relu,
          final_activation_fn=tf.nn.tanh,
          cost_type='l2_norm'):

    N, H, W, C = imgs.shape
    all_xs, all_ys = [], []
    for img_i, img in enumerate(imgs):
        xs, ys = split_image(img)
        all_xs.append(np.c_[xs, np.repeat(img_i, [xs.shape[0]])])
        all_ys.append(ys)
    xs = np.array(all_xs).reshape(-1, 3)
    xs = (xs - np.mean(xs, 0)) / np.std(xs, 0)
    xs = np.nan_to_num(xs)
    ys = np.array(all_ys).reshape(-1, 3)
    ys = ys / 127.5 - 1

    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model = build_model(xs, ys, n_neurons, n_layers,
                            activation_fn, final_activation_fn,
                            cost_type)
        optimizer = tf.train.AdamOptimizer(
            learning_rate=learning_rate).minimize(model['cost'])
        sess.run(tf.global_variables_initializer())
        costs = []
        resulting_imgs = None
        for i in range(n_iterations):
            # Get a random sampling of the dataset
            idxs = np.random.permutation(range(len(xs)))

            # The number of batches we have to iterate over
            n_batches = len(idxs) // batch_size
            training_cost = 0

            # Now iterate over our stochastic minibatches:
            for batch_i in range(n_batches):

                # Get just minibatch amount of data
                idxs_i = idxs[batch_i*batch_size: (batch_i + 1)*batch_size]

                # And optimize, also returning the cost so we can monitor
                # how our optimization is doing.
                cost = sess.run(
                    [model['cost'], optimizer],
                    feed_dict={model['X']: xs[idxs_i],
                               model['Y']: ys[idxs_i]})[0]
                training_cost += cost

            print('iteration {}/{}: cost {}'.format(i+1, n_iterations, training_cost/n_batches))

            costs.append(training_cost / n_batches)
        ys_pred = model['Y_pred'].eval(feed_dict={model['X']: xs}, session=sess)
        resulting_imgs = ys_pred.reshape(imgs.shape)
        return resulting_imgs


"""
Takes an input path to the images to train on.
"""
if __name__ == "__main__":
    dirname = sys.argv[1]

    filenames = [os.path.join(dirname, fname) for fname in os.listdir(dirname)]

    shape = (100, 100)
    imgs = [preprocess_file(f, shape) for f in filenames]
    imgs = np.array(imgs).copy()

    result = train(imgs, n_iterations=30)

    # last layer is tanh so make sure values are between 0-255
    result_gif = [np.clip(((m * 127.5) + 127.5), 0, 255).astype(np.uint8) for m in result]
    gif.build_gif(result_gif, saveto='{}.gif'.format(dirname))

