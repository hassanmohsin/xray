import matplotlib.pyplot as plt
import numpy as np


def exp_filter(data, n, forward=True):
    """
    To smooth out the signals. Lesser n makes lesser deviation in y-values
    :param data:
    :param n:
    :param forward:
    :return:
    """
    alpha = 1 / n
    beta = 1 - alpha
    data_f = np.zeros_like(data)
    if forward:
        data_f[0] = np.mean(data)
        for i in range(1, len(data)):
            data_f[i] = data_f[i - 1] * beta + data[i] * alpha
    return data_f


def draw_figs(objects, positions, max_height, base_size):
    """
    Plot figures
    :param objects:
    :param positions:
    :param max_height:
    :param base_size:
    :return:
    """
    np.random.seed(0)
    fig, ax = plt.subplots()
    for i in range(len(objects)):
        c = tuple(np.random.random(size=3))
        a, x1, off = positions[i][0], objects[i], positions[i][1]
        x = list(np.arange(a, a + x1.shape[1])) + list(np.arange(a + x1.shape[1] - 1, a - 1, -1))
        y = list(x1[0] - off) + list(x1[1, ::-1] - off)
        ax.fill(x, y, color=c)
    plt.xlim(0, base_size)
    plt.ylim(0, max_height)
    np.random.seed()
    plt.show()


if __name__ == '__main__':
    base_size = 101
    np.random.seed()
    x0 = exp_filter(np.random.normal(0, 1, base_size), 10)
    x0 = x0 - np.amin(x0)
    # x0 = np.zeros(base_size)  # Comment out for random choice of x0

    max_height = 50
    object_count = 200
    max_object_height = 20
    max_object_length = 25
    objects, positions = [], []
    display_steps = True
    plt.close('all')

    for i in range(object_count):
        # Create an arbitrarily-shaped 2D object
        len_x = np.random.randint(5, max_object_length)
        # x1[0] bottom surface, x1[1] top surface
        x1 = exp_filter(np.random.normal(0, max_object_height, (2, len_x)), base_size // 10)
        x1 = np.int32(np.sort(x1, axis=0))

        # Lower the bottom surface and connect the ends together n
        x1[0] -= 1
        x1[:, 0] = np.mean(x1[:, 0])
        x1[:, -1] = np.mean(x1[:, -1])

        # Find the shortest distances between the bottom surface and the ground at all possible
        #  position (sliding to the right and measuring)
        offsets = []
        for i in range(len(x0) - x1.shape[1] + 1):
            d = x1[0] - x0[i:i + x1.shape[1]]
            offset = np.amin(d)  # Shortest distance between the lowest part of the object (x1[0]) and the surface (x0)
            offsets.append(offset)

        # Find the position where the distance to the ground is maximum (lowest level of the ground)
        a = np.argmax(np.array(offsets))

        x0[a:a + x1.shape[1]] = x1[1] - offsets[a]
        if np.amax(x0) <= max_height:
            objects.append(x1)
            positions.append([a, offsets[a]])

            if len(objects) % 2 == 0:
                draw_figs(objects, positions, max_height, base_size)

    draw_figs(objects, positions, max_height, base_size)

    print('Number of objects', len(objects))
    print('Coverage:', sum([np.sum(x1[1] - x1[0]) for x1 in objects]) / max_height / (base_size - 1))
