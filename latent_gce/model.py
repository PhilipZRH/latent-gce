import time
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import latent_gce.networks as networks
from latent_gce.utils import batch_water_filling


class LatentGCEImage:
    def __init__(self,
                 env,  # gym environment
                 num_steps_observation,  # Number of steps to concatenate
                 action_raw_dim,  # Dimension of vector a
                 state_latent_dimension,  # Dimension of z
                 action_latent_dimension,  # Dimension of b
                 learning_rate=1e-4,
                 log_dir=None,
                 use_gpu=True,
                 load_model=None):
        self.env = env
        self.num_steps_observation = num_steps_observation
        self.graph = tf.Graph()
        with self.graph.as_default():
            if use_gpu and tf.test.is_gpu_available():
                device = '/device:GPU:0'
            else:
                device = '/cpu:0'
            with tf.device(device):
                self.obs_raw = tf.placeholder(shape=[None, 64, 64, self.num_steps_observation * 3],
                                              name="obs_raw",
                                              dtype=tf.float32)
                self.action_raw = tf.placeholder(shape=[None, action_raw_dim], name="action_raw", dtype=tf.float32)
                self.obs_raw_t = tf.placeholder(shape=[None, 64, 64, self.num_steps_observation * 3],
                                                name="obs_raw_t",
                                                dtype=tf.float32)

                # Encode to latent space
                self.obs_latent = networks.image_encoder(input_tensor=self.obs_raw,
                                                         latent_dim=state_latent_dimension,
                                                         name='obs_encoder')
                self.obs_t_latent = networks.image_encoder(input_tensor=self.obs_raw_t,
                                                           latent_dim=state_latent_dimension,
                                                           name='obs_encoder',
                                                           reuse=True)
                self.obs_decode = networks.image_decoder(input_tensor=self.obs_latent,
                                                         num_steps_observation=self.num_steps_observation,
                                                         name='obs_decoder')
                self.obs_t_decode = networks.image_decoder(input_tensor=self.obs_t_latent,
                                                           num_steps_observation=self.num_steps_observation,
                                                           name='obs_decoder',
                                                           reuse=True)

                self.action_latent = networks.build_mlp(self.action_raw,
                                                        action_latent_dimension,
                                                        name='action_encoder',
                                                        hidden_dims=(512, 512, 512))
                self.action_decode = networks.build_mlp(self.action_latent,
                                                        action_raw_dim,
                                                        name='action_decoder',
                                                        hidden_dims=(512, 512))

                self.A_linear = networks.build_mlp(self.obs_latent,
                                                   state_latent_dimension * action_latent_dimension,
                                                   name='matrix_A',
                                                   hidden_dims=(512, 512, 512))
                self.A_mat = tf.reshape(self.A_linear, (-1, state_latent_dimension, action_latent_dimension))
                self.singular_values = tf.linalg.svd(self.A_mat, compute_uv=False)
                self.predict = tf.linalg.matvec(self.A_mat, self.action_latent)
                self.predict_reconstruct = networks.image_decoder(input_tensor=self.predict,
                                                                  num_steps_observation=self.num_steps_observation,
                                                                  name='obs_decoder',
                                                                  reuse=True)

                self.latent_error = self.predict - self.obs_t_latent
                self.latent_prediction_loss = tf.reduce_mean(self.latent_error ** 2)

                self.predict_reconstruct_loss = tf.reduce_mean((self.obs_raw_t - self.predict_reconstruct) ** 2)
                self.l2_obs = tf.reduce_mean((self.obs_decode - self.obs_raw) ** 2)
                self.l2_action = tf.reduce_mean((self.action_decode - self.action_raw) ** 2)
                self.individual_loss = None

                self.avg_obs_latent_l2 = tf.reduce_mean(self.obs_latent ** 2)
                self.avg_action_latent_l2 = tf.reduce_mean(self.action_latent ** 2)
                self.latent_obs_regularization = tf.abs(1 - self.avg_obs_latent_l2)
                self.latent_action_regularization = tf.abs(1 - self.avg_action_latent_l2)
                self.latent_regularization = self.latent_obs_regularization + self.latent_action_regularization

                loss_list = [self.latent_prediction_loss,
                             5e2 * self.l2_obs,
                             5e2 * self.predict_reconstruct_loss,
                             10 * self.l2_action,
                             self.latent_regularization]
                self.loss = sum(loss_list)

                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
                self.initializer = tf.global_variables_initializer()

                tf.summary.scalar('Loss', self.loss)
                tf.summary.scalar('Latent Prediction Loss', self.latent_prediction_loss)
                tf.summary.scalar('Observation L2 Loss', self.l2_obs)
                tf.summary.scalar('Action L2 Loss', self.l2_action)
                tf.summary.scalar('Prediction Reconstruction Loss', self.predict_reconstruct_loss)
                tf.summary.scalar('Regularization', self.latent_regularization)
                self.merged = tf.summary.merge_all()
                if log_dir:
                    self.writer = tf.summary.FileWriter(log_dir, self.graph)
                else:
                    self.writer = None
                self.saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        if load_model:
            self.saver.restore(self.sess, load_model)
        else:
            self.sess.run(self.initializer)

    def train(self, train_data, batch_size=128, num_epoch=100, draw_img_frequency=0, draw_img_folder='draw',
              verbose=True, loss_cap=None):
        obs_data = train_data['obs']
        action_data = train_data['actions']
        obs_t_data = train_data['obs_t']

        num_samples = len(obs_data)
        idx = np.asarray(range(num_samples))
        num_batches = (num_samples - 1) // batch_size + 1

        if verbose:
            print('##############################')
            print('Start training...')
            print(obs_data.shape)
            print(action_data.shape)
            print(obs_t_data.shape)
            print('##############################')

        i = 1
        feed_dict = None
        iteration_loss = 0

        for epoch in range(num_epoch):
            time_start = time.time()
            iteration_loss = 0
            np.random.shuffle(idx)
            for bn in range(num_batches):
                idxes = idx[bn * batch_size: (bn + 1) * batch_size]
                feed_dict = {self.obs_raw: obs_data[idxes],
                             self.action_raw: action_data[idxes],
                             self.obs_raw_t: obs_t_data[idxes]}
                loss_value, _ = self.sess.run([self.loss, self.optimizer], feed_dict)
                iteration_loss += loss_value

            iteration_loss /= num_batches
            if draw_img_frequency and i % draw_img_frequency == 0:
                self.draw_enc_images(epoch, draw_img_folder)

            summary = self.sess.run(self.merged, feed_dict)
            if self.writer:
                self.writer.add_summary(summary, i)
            i += 1
            time_stop = time.time()
            time_delta = time_stop - time_start
            if verbose:
                print("epoch %d \t train loss %.6f \t time taken: %.1f" % (epoch, iteration_loss, time_delta))
            elif epoch % 10 == 0 and epoch != 0:
                print(epoch, end=' ', flush=True)

            if loss_cap and iteration_loss < loss_cap:
                break

        return iteration_loss

    def calculate_loss(self, data):
        obs_data = data['obs']
        action_data = data['actions']
        obs_t_data = data['obs_t']
        feed_dict = {self.obs_raw: obs_data,
                     self.action_raw: action_data,
                     self.obs_raw_t: obs_t_data}
        loss_value = self.sess.run(self.loss, feed_dict)
        return loss_value

    def draw_enc_images(self, epoch, folder):
        if not folder:
            return
        draw_img_org_list = [cv2.resize(self.env.reset(), dsize=(64, 64)) / 255]
        for i in range(self.num_steps_observation - 1):
            draw_img_org_list.append(cv2.resize(self.env.step([0])[0], dsize=(64, 64)) / 255)
        draw_img_org = [np.concatenate(draw_img_org_list, axis=2)]
        draw_feed = {self.obs_raw: draw_img_org}
        draw_img_predict = self.sess.run(self.obs_decode, draw_feed)
        actual_img = draw_img_org[0][:, :, :3]
        dec_img = draw_img_predict[0][:, :, :3]
        actual_img = actual_img.clip(0, 1)
        dec_img = dec_img.clip(0, 1)
        plt.imsave(folder + '/' + str(epoch) + '_actual.png', actual_img)
        plt.imsave(folder + '/' + str(epoch) + '_dec.png', dec_img)

    def draw_predict_images(self, start, action, epoch, folder):
        if not folder:
            return
        self.env.reset()
        self.env.unwrapped.state = start
        start_img_list = [cv2.resize(self.env.get_obs(), dsize=(64, 64)) / 255]
        for i in range(self.num_steps_observation - 1):
            start_img_list.append(cv2.resize(self.env.step([0])[0], dsize=(64, 64)) / 255)
        self.env.reset()
        self.env.unwrapped.state = start
        end_img_list = []
        for a in action:
            end_img_list = [cv2.resize(self.env.step([a])[0], dsize=(64, 64)) / 255]
        for i in range(self.num_steps_observation - 1):
            end_img_list.append(cv2.resize(self.env.step([0])[0], dsize=(64, 64)) / 255)
        start_img = [np.concatenate(start_img_list, axis=2)]
        end_img = [np.concatenate(end_img_list, axis=2)]
        feed_dict = {self.obs_raw: start_img,
                     self.action_raw: [action]}
        end_img_predict = self.sess.run(self.predict_reconstruct, feed_dict)
        actual_img = end_img[0][:, :, :3]
        pred_img = end_img_predict[0][:, :, :3]
        actual_img = actual_img.clip(0, 1)
        pred_img = pred_img.clip(0, 1)
        plt.imsave(folder + '/' + str(epoch) + '_actual.png', actual_img)
        plt.imsave(folder + '/' + str(epoch) + '_pred.png', pred_img)

    def forward_singular_values(self, observations, batch_size=1024):
        result = []
        for batch_idx in range((len(observations) - 1) // batch_size + 1):
            batch_result = self.sess.run(self.singular_values,
                                         feed_dict={self.obs_raw: observations[batch_idx * batch_size:
                                                                               (batch_idx + 1) * batch_size]})
            result.append(batch_result)
        result = np.concatenate(result, axis=0)
        return result

    def draw_showcase(self, obs, action):
        feed_dict = {self.obs_raw: [obs],
                     self.action_raw: [action]}
        pred = self.sess.run(self.predict_reconstruct, feed_dict)[0].clip(0, 1)
        return pred

    def water_filling_from_observations(self, observations, power=1, batch_size=1024):
        singular_values = self.forward_singular_values(observations, batch_size=batch_size)
        singular_values = list(singular_values)
        return np.array(batch_water_filling(singular_values, power=power))

    def water_filling_local_magnification(self, observations, magnification, power=1):
        singular_values = self.forward_singular_values(observations)
        singular_values = [s * magnification for s in singular_values]
        return batch_water_filling(singular_values, power=power)

    def save(self, folder):
        self.saver.save(self.sess, folder)


# Compute empowerment from raw action and observations. Assumes that they are 1D vectors.
class LatentGCEIdentity(LatentGCEImage):
    def __init__(self,
                 obs_raw_dim,  # Dimension of each observation
                 action_raw_dim,  # Dimension of vector a
                 learning_rate=1e-4,
                 obs_selection=None,
                 log_dir=None,
                 use_gpu=True,
                 load_model=None
                 ):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if use_gpu and tf.test.is_gpu_available():
                device = '/device:GPU:0'
            else:
                device = '/cpu:0'
            with tf.device(device):
                self.obs_raw = tf.placeholder(shape=[None, obs_raw_dim], name="obs_raw", dtype=tf.float32)
                self.action_raw = tf.placeholder(shape=[None, action_raw_dim], name="action_raw", dtype=tf.float32)
                self.obs_raw_t = tf.placeholder(shape=[None, obs_raw_dim], name="obs_raw_t", dtype=tf.float32)

                if obs_selection is None:
                    self.mtx_height = obs_raw_dim
                else:
                    self.mtx_height = len(obs_selection)
                A_input = self.obs_raw
                self.A_linear = networks.build_mlp(A_input,
                                                   self.mtx_height * action_raw_dim,
                                                   name='matrix_A',
                                                   hidden_dims=(512, 512, 512))
                self.A_mat = tf.reshape(self.A_linear, (-1, self.mtx_height, action_raw_dim))
                with tf.device('/cpu:0'):
                    self.singular_values = tf.linalg.svd(self.A_mat, compute_uv=False)
                self.bias = networks.build_mlp(self.obs_raw,
                                               self.mtx_height,
                                               name='bias',
                                               hidden_dims=(256, 256, 256))
                self.predict = tf.linalg.matvec(self.A_mat, self.action_raw) + self.bias
                if obs_selection is None:
                    self.ground_truth = self.obs_raw_t
                else:
                    self.ground_truth = tf.gather(self.obs_raw_t, tf.constant(obs_selection), axis=1)

                self.prediction_error = self.predict - self.ground_truth
                self.individual_loss = tf.reduce_mean(self.prediction_error ** 2, axis=-1)
                self.prediction_loss = tf.reduce_mean(self.individual_loss)

                self.loss = self.prediction_loss
                self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

                self.initializer = tf.global_variables_initializer()

                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()
                if log_dir:
                    self.writer = tf.summary.FileWriter(log_dir, self.graph)
                else:
                    self.writer = None
                self.saver = tf.train.Saver()

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.graph, config=config)
        if load_model:
            self.saver.restore(self.sess, load_model)
        else:
            self.sess.run(self.initializer)
