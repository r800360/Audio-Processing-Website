def buffer(input_data, sample_rate, window_size, hop_size):
    output = np.array([input_data[i:i+window_size] for i in range(0, len(input_data)-window_size, hop_size)])
    return output.T

def denoise_audio(file_path):
    # Load audio file using librosa
    x, fs = load(file_path, sr=None)

    x = x.astype(np.float32)
    if x.ndim > 1:
        x = np.mean(x, axis=1)  # Convert to mono if stereo

    xmat = buffer(x, fs, 400, 200)

    n_inputs = np.shape(xmat)[0]
    n_hidden = 2
    learning_rate = 0.01

    X = tf.compat.v1.placeholder(tf.float32, shape=[None, n_inputs])
    W = tf.Variable(tf.compat.v1.truncated_normal(stddev=.1, shape=[n_inputs, n_hidden]))

    hidden = tf.matmul(X, W)
    outputs = tf.matmul(hidden, tf.transpose(W))

    reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(reconstruction_loss)

    init = tf.compat.v1.global_variables_initializer()

    n_iterations = 10000
    X_train = xmat.T
    X_test = X_train

    col = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

    with tf.compat.v1.Session() as sess:
        init.run()
        for iteration in range(n_iterations):
            sess.run(training_op, feed_dict={X: X_train})

            if iteration % 1000 == 0:
                W_val = sess.run(W)
                # plt.clf()
                # for k in range(n_hidden):
                #     plt.subplot(n_hidden, 1, k + 1)
                #     plt.plot(W_val[:, k], col[k % len(col)])
                # plt.show(False)
                # plt.pause(0.001)

        codings_val = sess.run(hidden, feed_dict={X: X_test})
    
    # Reconstruct the denoised signal
    denoised_xmat = np.dot(codings_val, W_val.T)
    denoised_signal = np.zeros(len(x))
    for i in range(0, denoised_signal.shape[0] - 400, 200):
        denoised_signal[i:i+400] += denoised_xmat[:, i//200]
    
    denoised_signal = (denoised_signal * 32767).astype(np.int16)
    denoised_file_path = file_path.replace('.mp3', '_denoised.wav').replace('.wav', '_denoised.wav')
    wavfile.write(denoised_file_path, fs, denoised_signal)
    
    return denoised_file_path