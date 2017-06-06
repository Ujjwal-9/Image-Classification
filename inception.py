def inception2d(input, num_channels, num_filter):
    # bias dimension = 3*num_filter and then the extra num_channels for the avg pooling
    bias = tf.Variable(tf.random_normal([3*num_filter + num_channels]))

    # 1x1
    one_filter = tf.Variable(tf.random_normal([1, 1, num_channels, num_filter]))
    one_by_one = tf.nn.conv2d(input, one_filter, strides=[1, 1, 1, 1], padding='SAME')

    # 3x3
    three_filter = tf.Variable(tf.random_normal([3, 3, num_channels, num_filter]))
    three_by_three = tf.nn.conv2d(input, three_filter, strides=[1, 1, 1, 1], padding='SAME')

    # 5x5
    five_filter = tf.Variable(tf.random_normal([5, 5, num_channels, num_filter]))
    five_by_five = tf.nn.conv2d(input, five_filter, strides=[1, 1, 1, 1], padding='SAME')

    # avg pooling
    pooling = tf.nn.avg_pool(input, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME')

    input = tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)  # Concat in the 4th dim to stack
    input = tf.nn.bias_add(input, bias)
    return tf.nn.relu(input)