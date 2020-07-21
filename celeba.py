import os
import collections
import math
import pickle as pkl
import scipy.misc as misc
import numpy as np
import tensorflow as tf

# The model takes as input a `NPRegressionDescription` namedtuple
# with fields:
#   `query`: a tuple containing ((context_x, context_y), target_x)
#   `target_y`: a tensor containing the ground truth for the targets to be
#     predicted
#   `num_total_points`: A vector containing a scalar that
#     describes the total
#     number of datapoints used (context + target)
#   `num_context_points`: A vector containing a scalar that
#    describes the number
#    of datapoints used as context
# The GPCurvesReader returns the newly sampled data in this format at each
# iteration

NPRegressionDescription = collections.namedtuple(
    "NPRegressionDescription",
    ("query", "target_y", "num_total_points", "num_context_points",
     "hyperparams"))

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def download_celeba(save_dir):

    # download
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not os.path.exists(os.path.join(save_dir,'img_align_celeba.zip')):
        print("downloading celeba")
        file_id = '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
        destination = os.path.join(save_dir,'img_align_celeba.zip')
        download_file_from_google_drive(file_id, destination)

    if not os.path.exists(os.path.join(save_dir,'img_align_celeba')):
        print("unpacking celeba")
        zip_file = zipfile.ZipFile(destination)
        zip_file.extractall(save_dir)
        zip_file.close()

    if not os.path.exists(os.path.join(save_dir,'list_eval_partition.txt')):
        print("downloading celeba tr/val/test list")
        file_id = '0B7EVK8r0v71pY0NSMzRuSXJEVkk'
        destination = os.path.join(save_dir,'list_eval_partition.txt')
        download_file_from_google_drive(file_id, destination)

    with open(os.path.join(save_dir,'list_eval_partition.txt')) as f:
        file_list = np.array([line[:-1].split(' ') for line in f.readlines()])

    # get images
    tr_list = file_list[file_list[:,1]=='0']
    val_list = file_list[file_list[:,1]=='1']
    ts_list = file_list[file_list[:,1]=='2']
    val_list = np.concatenate([val_list, ts_list], axis=0)

    return tr_list, val_list

def get_image(save_dir, fname_list, obj_shape=(32,32), prefix='train'):

    pkl_name = prefix + '_' + str(obj_shape[0]) + '.pkl'
    if os.path.exists(os.path.join(save_dir, pkl_name)):
        with open(os.path.join(save_dir, pkl_name), 'rb')as f:
            images = pkl.load(f)
    else:
        images = []
        for f_idx, fname in enumerate(fname_list):
            if f_idx % 10000 == 0:
                print(str(f_idx) + "th resized")
            img = misc.imread(os.path.join(save_dir, 'img_align_celeba',
                                        fname[0]), mode='RGB')
            img = misc.imresize(img, size=obj_shape)
            images.append(img/255.0)

        images = np.array(images)

        with open(os.path.join(save_dir, pkl_name), 'wb') as f:
            pkl.dump(images, f)

    return images

class CelebaReader(object):
    """Generates images of moving Celeba.
    random face from training set of celeba moves random direction and speed in
    given speed range [speed_min, speed_max). At each time-step, we added
    a small Gaussian noise on the dynamics to modeling stochasticity.
    """

    def __init__(self,
                batch_size,
                max_num_context,
                testing=False,
                len_seq=10,
                len_given=5,
                len_gen=10,
                canvas_size=84,
                speed_min=2.0,
                speed_max=3.0,
                temporal=False,
                case=1,
                ):
        """__init__

        :param batch_size: batch size
        :param max_num_context: the maximum number of context
        :param testing: testing mode or not
        :param len_seq: FLAGS.LEN_SEQ
        :param len_given: FLAGS.LEN_GIVEN
        :param len_gen: FLAGS.LEN_GEN
        :param canvas_size: the entire image size
        :param speed_min: minimum value of the dynamics range
        :param speed_max: maximum value of the dynamics range
        :param temporal: temporal mode or not
        :param case: case option {1|2|3}
        """
        self._batch_size = batch_size
        self._max_num_context = max_num_context
        self._testing = testing
        self._len_seq = len_seq
        self._len_given = len_given
        self._len_gen = len_gen
        self._canvas_size = canvas_size
        self._speed_min = speed_min
        self._speed_max = speed_max
        self._temporal = temporal
        self._case = case
        self._noise_factor = 0.1
        self._sample_size = 32

        # download celeba dataset
        save_dir = 'celeba'
        tr_list, val_list = download_celeba(save_dir)

        if testing:
            images = get_image(save_dir, val_list,
                               (self._sample_size, self._sample_size), 'val')
            images = images[:5000]
        else:
            images = get_image(save_dir, tr_list,
                               (self._sample_size, self._sample_size), 'train')
            images = images[:80000]

        # normalization to [-0.5,0.5]
        images = images - 0.5
        self._images = tf.constant(images, dtype=tf.float32)
        self._num_samples = self._images.shape[0]

        x_values = []
        for i in range(canvas_size):
            for j in range(canvas_size):
                x_values.append([i,j])

        # normalization to [-1,1]
        x_values = np.array(x_values)
        x_values = 2 * (x_values / (canvas_size-1)) - 1
        self._x_values = tf.constant(np.array(x_values),dtype=tf.float32)

    def make_canvas(self, x_loc, y_loc, samples):
        """make_canvas, drawing face on black canvas
        """
        x_pad = tf.stack(
            [x_loc, self._canvas_size-(x_loc+self._sample_size)], axis=1)
        y_pad = tf.stack(
            [y_loc, self._canvas_size-(y_loc+self._sample_size)], axis=1)
        z_pad = tf.stack([x_loc*0,y_loc*0], axis=1)
        pad = tf.cast(tf.stack([x_pad, y_pad, z_pad], axis=1),dtype=tf.int32)
        canvas = []
        for b_idx in range(self._batch_size):
            canvas.append(tf.pad(samples[b_idx], pad[b_idx], 'CONSTANT',
                                constant_values=-0.5))
        canvas = tf.stack(canvas, axis=0)

        return canvas

    def bounce(self, loc, vel):
        """bounce, bouncing rule
        """
        loc_res, vel_res = [], []
        for b_idx in range(self._batch_size):
            loc_tmp = tf.cond(tf.less(loc[b_idx],0),
                                lambda: -1 * loc[b_idx],
                                lambda: loc[b_idx])
            vel_tmp = tf.cond(tf.less(loc[b_idx],0),
                                lambda: -1 * vel[b_idx],
                                lambda: vel[b_idx])
            loc_tmp = tf.cond(tf.greater_equal(
                loc[b_idx],self._canvas_size - self._sample_size),
                lambda: 2*(self._canvas_size-self._sample_size)-1*loc[b_idx],
                lambda: loc_tmp)
            vel_tmp = tf.cond(tf.greater_equal(
                loc[b_idx],self._canvas_size - self._sample_size),
                lambda: -1 * vel[b_idx],
                lambda: vel_tmp)

            loc_res.append(loc_tmp)
            vel_res.append(vel_tmp)

        loc = tf.stack(loc_res, axis=0)
        vel = tf.stack(vel_res, axis=0)

        return loc, vel

    def generate_temporal_curves(self, seed=None):
        """generate_temporal_curves, making a sequence of moving mnist
        context/target set
        """
        # Select samples
        idx = tf.random_shuffle(tf.range(self._num_samples), seed=seed)
        idx = idx[:self._batch_size]
        samples = tf.gather(self._images, idx, axis=0)

        # initial locations
        if self._canvas_size == self._sample_size:
            x_loc = tf.constant(np.zeros(self._batch_size), dtype=tf.float32)
            y_loc = tf.constant(np.zeros(self._batch_size), dtype=tf.float32)
        else:
            x_loc = tf.random_uniform([self._batch_size],
                                0, self._canvas_size-self._sample_size,
                                seed=seed, dtype=tf.float32)
            y_loc = tf.random_uniform([self._batch_size],
                                0, self._canvas_size-self._sample_size,
                                seed=seed, dtype=tf.float32)

        # Set dynamics
        if self._speed_min == self._speed_max:
            speed = tf.constant(self._speed_min * np.ones(self._batch_size),
                                dtype=tf.float32)
        else:
            speed = tf.random_uniform([self._batch_size],
                                self._speed_min, self._speed_max, seed=seed)
        direc = tf.random_uniform([self._batch_size], 0.0, 2*math.pi,
                                seed=seed)
        y_vel = speed * tf.math.sin(direc)
        x_vel = speed * tf.math.cos(direc)

        # initial canvas
        y_loc_int = tf.cast(y_loc, dtype=tf.int32)
        x_loc_int = tf.cast(x_loc, dtype=tf.int32)
        canvas = self.make_canvas(x_loc_int, y_loc_int, samples)

        curve_list = []
        if (self._case==2) or (self._case==3):
            # sparse time or long term tracking
            idx = tf.random_shuffle(tf.range(self._len_seq),
                                            seed=seed)[:(self._len_given)]
        for t in range(self._len_seq):
            if seed is not None:
                _seed = seed * t
            else:
                _seed = seed
            if self._case==1:    # using len_given
                if t < self._len_given:
                    num_context = tf.random_uniform(shape=[], minval=5,
                                maxval=self._max_num_context, dtype=tf.int32,
                                seed=_seed)
                else:
                    num_context = tf.constant(0)
            if self._case==2:    # sparse time
                nc_cond = tf.where(tf.equal(idx,t))
                nc_cond = tf.reshape(nc_cond, [-1])
                num_context = tf.cond(tf.equal(tf.size(nc_cond),0),
                            lambda:tf.constant(0),
                            lambda:tf.random_uniform(shape=[], minval=5,
                                                maxval=self._max_num_context,
                                                dtype=tf.int32, seed=_seed))
            if self._case==3:    # long term tracking
                nc_cond = tf.where(tf.equal(idx,t))
                nc_cond = tf.reshape(nc_cond, [-1])
                num_context = tf.cond(tf.equal(tf.size(nc_cond),0),
                            lambda:tf.constant(0),
                            lambda:tf.constant(30))

            if self._temporal:
                encoded_t = None
            else:
                encoded_t = 0.25 + 0.5*t/self._len_seq
            curve_list.append(self.generate_curves(canvas, num_context,
                                                _seed, encoded_t))
            vel_noise = y_vel * self._noise_factor * tf. random_normal(
                [self._batch_size], seed=_seed)
            y_loc += y_vel + vel_noise
            y_loc, y_vel = self.bounce(y_loc, y_vel)
            vel_noise = x_vel * self._noise_factor *  tf. random_normal(
                [self._batch_size], seed=_seed)
            x_loc += x_vel + vel_noise
            x_loc, x_vel = self.bounce(x_loc, x_vel)

            y_loc_int = tf.cast(y_loc, dtype=tf.int32)
            x_loc_int = tf.cast(x_loc, dtype=tf.int32)
            canvas = self.make_canvas(x_loc_int, y_loc_int, samples)

        if self._testing:
            for t in range(self._len_seq,self._len_seq+self._len_gen):
                if seed is not None:
                    _seed = seed * t
                else:
                    _seed = seed
                num_context = tf.constant(0)

                if self._temporal:
                    encoded_t = None
                else:
                    encoded_t = 0.25 + 0.5*t/self._len_seq
                curve_list.append(self.generate_curves(canvas,
                                                num_context,
                                                _seed,
                                                encoded_t))
                vel_noise = y_vel * self._noise_factor *  tf.random_normal(
                    [self._batch_size], seed=_seed)
                y_loc += y_vel + vel_noise
                y_loc, y_vel = self.bounce(y_loc, y_vel)
                vel_noise = x_vel * self._noise_factor *  tf.random_normal(
                    [self._batch_size], seed=_seed)
                x_loc += x_vel + vel_noise
                x_loc, x_vel = self.bounce(x_loc, x_vel)

                y_loc_int = tf.cast(y_loc, dtype=tf.int32)
                x_loc_int = tf.cast(x_loc, dtype=tf.int32)
                canvas = self.make_canvas(x_loc_int, y_loc_int, samples)

        context_x_list, context_y_list = [], []
        target_x_list, target_y_list = [], []
        num_total_points_list = []
        num_context_points_list = []
        for t in range(len(curve_list)):
            (context_x, context_y), target_x = curve_list[t].query
            target_y = curve_list[t].target_y
            num_total_points_list.append(curve_list[t].num_total_points)
            num_context_points_list.append(curve_list[t].num_context_points)
            context_x_list.append(context_x)
            context_y_list.append(context_y)
            target_x_list.append(target_x)
            target_y_list.append(target_y)

        query = ((context_x_list, context_y_list), target_x_list)

        return NPRegressionDescription(
                query=query,
                target_y=target_y_list,
                num_total_points=num_total_points_list,
                num_context_points=num_context_points_list,
                hyperparams=[tf.constant(0)])

    def generate_curves(self, canvas, num_context=3,
                        seed=None, encoded_t=None):
        """generate_curves

        :param canvas: taget canvas what we want to extract context and target
        :param num_context: the number of context in this time-step
        :param seed: random seed
        :param encoded_t: when temporal=True, time is encoded in query
        """

        # If we are testing we want to have more targets and have them evenly
        # distributed in order to plot the function.

        num_total_points = self._canvas_size * self._canvas_size
        if self._testing:
            num_target = num_total_points
        else:
            maxval = self._max_num_context - num_context + 1
            num_target = tf.random_uniform(shape=(), minval=1,
                                        maxval=maxval,
                                        dtype=tf.int32, seed=seed)
        x_values = tf.tile(
            tf.expand_dims(self._x_values,
                            axis=0),[self._batch_size, 1, 1])

        # [batch_size, num_total_points, 3]
        y_values = tf.reshape(canvas, [self._batch_size, num_total_points, 3])

        if self._testing:

            # Select the targets
            target_x = x_values
            target_y = y_values

            if encoded_t is not None:
                target_x = tf.concat([
                    target_x,
                    tf.ones([self._batch_size, num_total_points, 1])*encoded_t
                    ], axis=-1)

            # Select the observations
            idx = tf.random_shuffle(tf.range(num_target), seed=seed)
            context_x = tf.gather(x_values, idx[:num_context], axis=1)
            context_y = tf.gather(y_values, idx[:num_context], axis=1)

            if encoded_t is not None:
                context_x = tf.concat([
                    context_x,
                    tf.ones([self._batch_size, num_context, 1]) * encoded_t
                    ], axis=-1)

        else:
            # Select the targets which will consist of the context points
            # as well as some new target points
            idx = tf.random_shuffle(tf.range(num_total_points), seed=seed)
            target_x = tf.gather(x_values, idx[:num_target + num_context],
                                 axis=1)
            target_y = tf.gather(y_values, idx[:num_target + num_context],
                                 axis=1)

            if encoded_t is not None:
                target_x = tf.concat([
                    target_x,
                    tf.ones([self._batch_size, num_target + num_context, 1])
                            * encoded_t], axis=-1)

            # Select the observations
            context_x = tf.gather(x_values, idx[:num_context], axis=1)
            context_y = tf.gather(y_values, idx[:num_context], axis=1)

            if encoded_t is not None:
                context_x = tf.concat([
                    context_x,
                    tf.ones([self._batch_size, num_context, 1]) * encoded_t
                    ], axis=-1)

        query = ((context_x, context_y), target_x)

        return NPRegressionDescription(
            query=query,
            target_y=target_y,
            num_total_points=tf.shape(target_x)[1],
            num_context_points=num_context,
            hyperparams=[tf.constant(0)])
