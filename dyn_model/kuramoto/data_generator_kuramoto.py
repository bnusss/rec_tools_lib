"""Python class for Kuramoto generating, visualization."""

import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from network_of_oscillators_Kuramoto_model import *
from network_of_oscillators_integrate_and_sync_quantifiers import *
# import PIL
from io import BytesIO
import argparse





def plot_data(data_show, type='loc'):
    # data_show = np.transpose(data_show, [0, 3, 1, 2])
    color_nri = ['r', 'b', 'g', 'purple', 'orange', 'r', 'b', 'g', 'purple', 'orange', 'r']
    
    # print('plot_data', data_show.shape)

    if type == 'loc':
        data_show = data_show[:,:,0]
        print(data_show)
    elif type == 'vel':
        data_show = data_show[:,:,1]
    else:
        print("Wrong type: %s, type should be 'loc' or 'vel'" % type)
        exit(-1)

    # print('data_show', data_show.shape)

    t_show = np.arange(data_show.shape[1])
    node_num = data_show.shape[0]
    for node in range(node_num):
        plt.subplot(node_num, 1, node+1)
        plt.plot(t_show, data_show[node], color_nri[node], lw=3)


    buffer_ = BytesIO()
    plt.savefig(buffer_, format="png")
    buffer_.seek(0)
    image = PIL.Image.open(buffer_)
    # ar = np.asarray(image)
    # TODO
    # buffer_.close()
    del buffer_

    # Warning
    plt.close()

    return image

    # plt.imshow(ar)
    # plt.show()


def kc(adj):
    """
    Critical value of coupling constant depending on adj.
    l: largest eigenvalue of adj
    pc: g(0) or g(center)
    """
    p_c = 0.1111111111111
    a,b=np.linalg.eig(adj)
    k_0 = 2/(np.pi*p_c)
    l = a[0]
    
    k = k_0/l
    # print(k)
    return k



class Kuramoto(object):
    """
    
    If the adjacency matrix is not given, it will be generated randomly. like
        
        [[0. 0. 0. 1. 0.]
         [0. 0. 1. 1. 0.]
         [0. 1. 0. 0. 1.]
         [1. 1. 0. 0. 1.]
         [0. 0. 1. 1. 0.]]

    >>> kuramoto = Kuramoto(adj_fp, train_fp, test_fp, node_num=11, is_generate=True, train_len=100)
    
    Parameter combination (t_end=10, dt=0.01, ds=10) make the dataset visualition similar to
    Figure.5 and Figure.9 in NRI. the total data point is 100 after downsampling by 10.
    ds for downsample

    if ZeroDivisionError occured, run again.

    """

    def __init__(self, adj_fp, train_fp, val_fp, test_fp, k_over_kc, load_adj_fp=None, adj='no', is_generate=False,
                 node_num=5, steps_per_group=100,
                 t_end=10, dt=0.01, ds=10,
                 train_len=5, val_len=1,test_len=1):
        self.k_over_kc = k_over_kc
        self.load_adj_fp = load_adj_fp 
        self.adj_fp = adj_fp
        self.train_fp = train_fp
        self.val_fp = val_fp
        self.test_fp = test_fp
        self.node_num = node_num
        self.steps_per_group = steps_per_group
        self.t_end = t_end
        self.dt = dt
        self.ds = ds
        self.train_len = train_len
        self.val_len = val_len
        self.test_len = test_len

        if is_generate == False:
            self.load_data()
        else:
            self.adj, self.edge_arr = self._gen_edge_arr(adj,n_type=args.n_type)
            self.train_data,self.val_data, self.test_data = self._gen_data()
            self._dump_data()

    def _gen_edge_arr(self, adj, n_type = "ER"):
        """Generate edge array for kuramoto lib."""
        if adj == 'load':
            edge_list = []
            adj = np.load(self.load_adj_fp)
            for i in range(self.node_num):
                for j in range(self.node_num):
                    if adj[i][j] == 1:
                        edge_list.append([i,j])
            edge_arr = np.asarray(edge_list)

        elif adj == 'no':
            if n_type == 'ER':
                ER = nx.Graph()
                # add nodes
                for i in range(self.node_num):
                    ER.add_node(i,value = np.random.randint(0,1))
                # num of edges
                edges = []
                ER = nx.random_graphs.erdos_renyi_graph(self.node_num, args.ER_p)

                adj = nx.adjacency_matrix(ER).toarray()
                
                edges = ER.edges()
                edges = np.asarray(edges)
                edges = edges.tolist()
                
                for i in range(len(edges)):
                    edge = edges[i]   
                    edges.append([edge[1],edge[0]])
                edge_arr = np.asarray(edges)
            
            if n_type == 'nk':
                dg = nx.Graph()
                # add nodes
                for i in range(self.node_num):
                    dg.add_node(i,value = np.random.randint(0,1))
                edges = []
                dg = nx.random_graphs.random_regular_graph(args.average_degree, self.node_num)
                adj = nx.adjacency_matrix(dg).toarray()
                
                edges = dg.edges()
                edges = np.asarray(edges)
                edges = edges.tolist()
                
                for i in range(len(edges)):
                    edge = edges[i]   
                    edges.append([edge[1],edge[0]])
                edge_arr = np.asarray(edges)

            if n_type == 'BA':
                BA = nx.Graph()
                # add nodes
                for i in range(self.node_num):
                    BA.add_node(i,value = np.random.randint(0,1))
                # num of edges
                edges = []
                BA = nx.random_graphs.barabasi_albert_graph(self.node_num, args.BA_degree)

                adj = nx.adjacency_matrix(BA).toarray()
                
                edges = BA.edges()
                edges = np.asarray(edges)
                edges = edges.tolist()
                
                for i in range(len(edges)):
                    edge = edges[i]   
                    edges.append([edge[1],edge[0]])
                edge_arr = np.asarray(edges)

            if n_type == 'WS':
                WS = nx.Graph()
                # add nodes
                for i in range(self.node_num):
                    WS.add_node(i,value = np.random.randint(0,1))
                # num of edges
                edges = []
                WS = nx.random_graphs.watts_strogatz_graph(self.node_num, args.WS_neighbour, args.WS_p)

                adj = nx.adjacency_matrix(WS).toarray()
                
                edges = WS.edges()
                edges = np.asarray(edges)
                edges = edges.tolist()
                
                for i in range(len(edges)):
                    edge = edges[i]   
                    edges.append([edge[1],edge[0]])
                edge_arr = np.asarray(edges)

            
        print(adj)
        return adj, edge_arr

    def _gen_kuramoto_data(self):
        """Generate location and velocity data."""

        time_list = np.arange(0, self.steps_per_group, 1)
        
        # Parameter scope is the same as NRI paper.
        # Intrinsic frequencies
        w = np.random.uniform(1, 10, self.node_num)
        theta0 = np.random.uniform(0, 2*np.pi, self.node_num)
    
        c = kc(self.adj)*args.k_over_kc
        
        kuramoto_model = KuramotoModel(w, self.edge_arr, c)

        # s=0, for no skip
        out = integrate_and_measure(kuramoto_model, theta0, tf=self.t_end, h=self.dt, s=0)
        t, theta, r, psi = unpack_print(out)
        theta_ds = theta[::self.ds, :]

        loc_list = []
        vel_list = []

        for node in range(self.node_num):
            vel = np.diff(theta_ds[:,node])/self.dt
            # Be aware of sin.
            loc = np.sin(theta_ds[:-1,node])

            loc_list.append(loc)
            vel_list.append(vel)

            # t_show = np.arange(loc.shape[0])
            # plt.subplot(self.node_num, 1, node+1)
            # plt.plot(t_show, loc, self.color_nri[node], lw=3)
            # plt.plot(t_show, vel, self.color_nri[node], lw=3)

        # plt.show()

        loc_arr = np.array(loc_list)
        vel_arr = np.array(vel_list)
        # (5, 100, 2)
        loc_vel = np.stack((loc_arr, vel_arr), axis=-1)

        return loc_vel

    def _gen_data(self):
        def _gen_data_n_groups(n,type):
            data_n_groups = []
            for i in range(n):
                if i % 10 == 0:
                    print('generating '+type+' iter'+str(i))
                loc_vel = self._gen_kuramoto_data()
                data_n_groups.append(loc_vel)
            return np.stack(data_n_groups, axis=0)

        train_data = _gen_data_n_groups(self.train_len,'train')
        val_data = _gen_data_n_groups(self.val_len,'val')
        test_data = _gen_data_n_groups(self.test_len,'test')

        # Reshape to: [num_sims, num_timesteps , num_dims,num_atoms]
        # train_data = np.transpose(train_data, [0, 2, 3, 1])
        # val_data = np.transpose(val_data, [0, 2, 3, 1])
        # test_data = np.transpose(test_data, [0, 2, 3, 1])

        return train_data,val_data, test_data

    def _dump_data(self):
        # train_data.shape (4, 5, 100, 2)
        # test_data.shape (1, 5, 100, 2)
        print('train_data.shape', self.train_data.shape)
        print('val_data.shape', self.val_data.shape)
        print('test_data.shape', self.test_data.shape)

        np.save(self.adj_fp, self.adj)
        np.save(self.train_fp, self.train_data)
        np.save(self.val_fp, self.val_data)
        np.save(self.test_fp, self.test_data)

    def load_data(self):
        self.adj = np.load(self.adj_fp)
        self.train_data = np.load(self.train_fp)
        self.val_data = np.load(self.val_fp)
        self.test_data = np.load(self.test_fp)
        self.node_num = self.adj.shape[0]
        # plot_data()
        return self.adj, self.train_data, self.val_data, self.test_data

    def load_adj(self):
        self.adj = np.load(self.load_adj_fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-train', type=int, default=50,help='reinit time in train set')
    parser.add_argument('--num-valid', type=int, default=10,help='reinit time in valid set')
    parser.add_argument('--num-test', type=int, default=10,help='reinit time in test set')
    parser.add_argument('--n_type', type=str, default='ER',help='type of network')
    parser.add_argument('--num-node', type=int, default=10,help='num of node')
    parser.add_argument('--ER_p', type=int, default=0.5,help='possibility of connection')
    parser.add_argument('--average_degree', type=int, default=4,help='average degree of nodes')
    parser.add_argument('--BA_degree', type=int, default=2,help='num edges to add at a time')
    parser.add_argument('--WS_neighbour', type=int, default=4,help='num neighbours')
    parser.add_argument('--WS_p', type=int, default=0.3,help='probability of reconnection')
    parser.add_argument('--k_over_kc', type=float, default=1.1, help='k/kc')

    args = parser.parse_args()
    
    k_over_kc = args.k_over_kc
    # address to save data
    load_fp = './'+str(args.n_type)+'adj-'+str(args.num_node)+'sample-'+str(args.average_degree)+'degree'+str(args.num_node)+'node-100timestep-2vec.npy'
    train_fp = './'+str(args.n_type)+'train-'+str(args.num_train)+'sample-'+str(args.k_over_kc)+'kc'+str(args.num_node)+'node-100timestep-2vec.npy'
    val_fp = './'+str(args.n_type)+'val-'+str(args.num_valid)+'sample-'+str(args.k_over_kc)+'kc'+str(args.num_node)+'node-100timestep-2vec.npy'
    test_fp = './'+str(args.n_type)+'test-'+str(args.num_test)+'sample-'+str(args.k_over_kc)+'kc'+str(args.num_node)+'node-100timestep-2vec.npy'
    adj_fp = './'+str(args.n_type)+'adj-'+str(args.num_node)+'sample-'+str(args.k_over_kc)+'kc'+str(args.num_node)+'node-100timestep-2vec.npy'
    kuramoto = Kuramoto(adj_fp, train_fp,val_fp, test_fp, k_over_kc, load_adj_fp=load_fp, adj='no', node_num=args.num_node, is_generate=True, train_len=args.num_train,val_len=args.num_valid, test_len=args.num_test)

    # kuramoto = Kuramoto(adj_fp, train_fp, test_fp, is_generate=False)
    # adj, train_data, test_data = kuramoto.load_data()
    # print('adj.shape', adj.shape)
    # print('adj:\n', adj)
    # print('train_data.shape', train_data.shape)
    # print('test_data.shape', test_data.shape)
    # data = np.load(train_fp)
    # plot_data(data,'loc')