import os
import sys
import torch
import networkx as nx
import numpy as np
import torchdiffeq as ode
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

# Same as Moving MNIST
width = 64
height = 64
assert width == height
nodes = width * height
num_objs = 1
batch_size = 2
t = torch.linspace(0., 1.0, 11).to(float)

data_dir = '/data/wangshuo/data/heat_diffusion'
fp = os.path.join(data_dir, 'heat_diffusion_batch_%d_size_%dx%d_objs_%d.npy' % (batch_size, width, height, num_objs))


class HeatDiffusion(nn.Module):
    # In this code, row vector: y'^T = y^T A^T      textbook format: column vector y' = A y
    def __init__(self, L, k=1):
        super(HeatDiffusion, self).__init__()
        self.L = -L  # Diffusion operator
        self.k = k  # heat capacity

    def forward(self, t, x):
        """
        :param t:  time tick
        :param x:  initial value:  is 2d row vector feature, n * dim
        :return: dX(t)/dt = -k * L *X
        If t is not used, then it is autonomous system, only the time difference matters in numerical computing
        """

        # print(t.cpu().data.numpy())

        f = torch.mm(self.L, x)
        return self.k * f


def grid_8_neighbor_graph(N):
    """
    Build discrete grid graph, each node has 8 neighbors
    :param n:  sqrt of the number of nodes
    :return:  A, the adjacency matrix
    """
    N = int(N)
    n = int(N ** 2)
    dx = [-1, 0, 1, -1, 1, -1, 0, 1]
    dy = [-1, -1, -1, 0, 0, 1, 1, 1]
    A = torch.zeros(n, n)
    for x in range(N):
        for y in range(N):
            index = x * N + y
            for i in range(len(dx)):
                newx = x + dx[i]
                newy = y + dy[i]
                if N > newx >= 0 and N > newy >= 0:
                    index2 = newx * N + newy
                    A[index, index2] = 1
    return A.float()


def gen_dataset():
    print("Generate heat diffusion data:\nHeight=width=%d, %d nodes." % (width, nodes))
    A = grid_8_neighbor_graph(width)
    G = nx.from_numpy_array(A.numpy())
    D = torch.diag(A.sum(1))
    L = (D - A)
    L = L.float()

    data_batch = []

    for i in tqdm(range(batch_size)):
        arr = np.zeros((width, width))

        dyn_k = np.random.randint(8, 15)

        for j in range(num_objs):
            init_size = np.random.randint(5, 15)
            init_value = np.float32(np.random.randint(128, 255))
            cord_x0 = np.random.randint(width - init_size)
            cord_y0 = np.random.randint(width - init_size)
            arr[cord_x0: cord_x0 + init_size, cord_y0: cord_y0 + init_size] = init_value
        arr = torch.from_numpy(arr).view(-1, 1).float()
        with torch.no_grad():
            solution_numerical = ode.odeint(HeatDiffusion(L, dyn_k), arr, t, method='dopri5')   # shape: 1000 * 1 * 2
        data_batch.append(solution_numerical)
    data_batch = torch.stack(data_batch, dim=1)

    imgs_dataset = []
    for j in tqdm(range(batch_size)):
        imgs_t = []
        for i, t0 in enumerate(t):
            if i == 0: continue
            imgs = data_batch[i][j, :].numpy()
            imgs = np.reshape(imgs, (width, width))
            imgs_t.append(imgs)
            plt.imshow(imgs)
            plt.show()
        sys.exit()
        imgs_t = np.stack(imgs_t, axis=0)
        imgs_dataset.append(imgs_t)
    imgs_dataset = np.stack(imgs_dataset, axis=0)
    print('dataset shape:', imgs_dataset.shape)
    np.save(fp, imgs_dataset)


def load_dataset():
    fp = '/data/wangshuo/data/heat_diffusion/heat_diffusion_batch_1_size_64x64_objs_1.npy'
    imgs_dataset = np.load(fp)
    print(imgs_dataset.shape)

    batch = 0
    for i in range(imgs_dataset.shape[1]):
        imgs = imgs_dataset[batch, i, :]
        plt.imshow(imgs)
        plt.show()


def main():
    gen_dataset()
    # load_dataset()


if __name__ == '__main__':
    main()
