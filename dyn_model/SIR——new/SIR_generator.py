import numpy as np
from scipy import *
import pandas as pd
import datetime
import pickle
import time
from scipy.integrate import odeint
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--times', type=int, default=10000, help='一共采样多少次, default=5000')
parser.add_argument('--sample_freq', type=int, default=50, help='采样频率 50步踩1次')
args = parser.parse_args()

# 引入 城市数据
df1 = pd.read_csv('./data/county_city_province.csv')
# 读取城市人口数据
# cities1存储所有城市的名称
df2 = pd.read_csv('./data/citypopulation.csv')
cities1 = set((df1['CITY']))
cities2 = set((df2['city']))
# 城市取并集
cities = set(list(cities1) + list(cities2))
nodes = {}
city_properties = {}
id_city = {}
for ct in cities:
    nodes[ct] = len(nodes)
    city_properties[ct] = {'pop': 1, 'prov': '', 'id': -1}
for i in df2.iterrows():
    # 城市名称和城市人口数对应上
    city_properties[i[1][0]] = {'pop': float(i[1][1])}
for i in df1.iterrows():
    # dict.get(key, default=None)
    # 如果指定值不在则返回该默认值
    # 完善城市信息的字典 key;城市名 value是{'pop':,'prov':,'id':}
    dc = city_properties.get(i[1]['CITY'], {})
    dc['prov'] = i[1]['PROV']
    dc['id'] = i[1]['CITY_ID']
    city_properties[i[1]['CITY']] = dc
    # id 和城市名称的一个应声 key id value 是城市名
    id_city[dc['id']] = i[1]['CITY']

def flushPrint(variable):
    sys.stdout.write('\r')
    sys.stdout.write('%s' % variable)
    sys.stdout.flush()


def generate_network():
    # 记录流量i->j
    df = pd.read_csv('./data/city_flow_v1.csv')
    flows = {}
    for n, i in enumerate(df.iterrows()):
        if n % 1000 == 0:
            flushPrint(n / len(df))
        # cityi = long(i[1]['cityi.id'])
        # cityj = long(i[1]['cityj.id'])
        cityi = (i[1]['cityi.id'])
        cityj = (i[1]['cityj.id'])
        value = flows.get((cityi, cityj), 0)
        flows[(cityi, cityj)] = value + i[1]['flowij']
        if cityi == 341301:
            print(flows[(cityi, cityj)])

    # save to flux matrix
    matrix = np.zeros([len(nodes), len(nodes)])
    self_flux = np.zeros(len(nodes))
    pij1 = np.zeros([len(nodes), len(nodes)])
    for key, value in flows.items():
        id1 = nodes.get(id_city[key[0]], -1)
        id2 = nodes.get(id_city[key[1]], -1)
        matrix[id1, id2] = value
    for i in range(matrix.shape[0]):
        self_flux[i] = matrix[i, i]
        matrix[i, i] = 0
        if np.sum(matrix[i, :]) > 0:
            pij1[i, :] = matrix[i, :] / np.sum(matrix[i, :])

    df = pd.read_csv('./data/Pij_BAIDU.csv', encoding='gbk')
    # 百度数据的迁徙？
    cities = {d: i for i, d in enumerate(df['Cities'])}
    pij2 = np.zeros([len(nodes), len(nodes)])
    for k, ind in cities.items():
        row = df[k]
        for city, column in cities.items():
            i_indx = nodes.get(city, -1)
            if i_indx < 0:
                print(city)
            j_indx = nodes.get(k, -1)
            if j_indx < 0:
                print(k)
            if i_indx >= 0 and j_indx >= 0:
                pij2[j_indx, i_indx] = row[column] / 100
                if i_indx == j_indx:
                    pij2[i_indx, j_indx] = 0
    bools = pij2 <= 0
    pij = np.zeros([pij1.shape[0], pij1.shape[0]])
    for i in range(pij1.shape[0]):
        row = pij1[i]
        bool1 = bools[i]
        values = row * bool1
        if np.sum(values) > 0:
            ratios = values / np.sum(values)
            sum2 = np.sum(pij2[i, :])
            pij[i, :] = (1 - sum2) * ratios + pij2[i, :]
    # np.argwhere 返回满足条件的索引
    zeros = np.argwhere(np.sum(pij, axis=1) == 0).reshape(-1)
    for idx in zeros:
        pij[idx][idx] = 1

    print(np.sum(pij, 1))  # Testing normalization
    # 大于0.00874037469 设置为1 4%
    # 大于0.054022140660.设置为1 1%
    # 0.02210023336 2%
    #         #     # 大于0.00796352246的设置为1 4%
    pij_c = (pij > 0.02210023336) + 0
    pij_c = pij_c.astype(int)
    # return pij
    return pij_c.T
#
def diff(sicol, t, r_0, t_l, gamma, pijt):
    # sicol:感染、确诊、易感
    sz = sicol.shape[0] // 3
    Is = sicol[:sz]  # infected 感染
    Rs = sicol[sz:2 * sz]  # 康复
    ss = sicol[2 * sz:]  # susceptible 易感

    # I_term = pijt.dot(Is) - Is * np.sum(pijt, axis=1)
    # S_term = pijt.dot(ss) - ss * np.sum(pijt, axis=1)
    # R_term = pijt.dot(Rs) - Rs * np.sum(pijt, axis=1)

    I_term = Is.dot(pijt) - Is * np.sum(pijt, axis=0)
    S_term = ss.dot(pijt) - ss * np.sum(pijt, axis=0)
    R_term = Rs.dot(pijt) - Rs * np.sum(pijt, axis=0)
    cross_term = r_0 * Is * ss / t_l

    delta_I = cross_term - Is / t_l + gamma * I_term
    delta_S = - cross_term + gamma * S_term
    deta_R = Is / t_l + gamma * R_term
    output = np.r_[delta_I, deta_R, delta_S]
    return output


#生成模拟数据
def generate_data(matrix):

    df = pd.read_csv('./data/R_cases_cum.csv', encoding='gbk')
    wuhan = df.loc[df['city_name'] == '武汉市', ['confirm', 'time', 'heal', 'dead']]
    dates = list(wuhan['time'])
    sorted_dates = np.sort(dates)
    first_date = datetime.datetime(2020, 1, 1, 0, 0)
    first_cases = int(wuhan.loc[wuhan['time'] == '2020-01-01']['confirm'])
    experiments = pd.read_pickle('./parameters/experiments_ti_tr_120_new.pkl')
    experiments = experiments + pd.read_pickle('./parameters/experiments_ti_tr_120_new_2.pkl')
    experiments = experiments + pd.read_pickle('./parameters/experiments_ti_tr_120_new_3.pkl')
    best_para = experiments[0]
    fit_param = sorted([(vvv[1], i) for i, vvv in enumerate(best_para)])
    itm = best_para[fit_param[0][1]]

    t_days = 300  # 200 #300

    steps = 1000  # 1000
    r0 = itm[0][0][0].item()
    initial_latent = itm[0][0][1].item()
    t_l = itm[2][1]  # 8.89 #一个病患潜伏期时间，参考SARS、MERS
    gamma = itm[2][2]  # 0.03#flowing_ratio

    timespan = np.linspace(0, t_days, steps)

    # r0 = 2.3
    # t_l = 8.5
    # gamma = 0.2
    #
    # timespan = np.linspace(0, t_days, steps)


    for i in range(args.times):
        if i % 100 ==0:
            print(10*i)

        Is0 = np.zeros(len(nodes))
        Ss0 = np.ones(len(nodes))
        Rs0 = np.zeros(len(nodes))

        city_chose = random.randint(0, 371)
        # print(city_chose)
        Is0[city_chose] = random.uniform(1, 10) * (1e-4)
        Rs0[city_chose] = random.uniform(1, 10) * (1e-4)
        Ss0[city_chose] = 1 - Is0[city_chose] - Rs0[city_chose]

        # Is0[nodes['武汉市']] = float(initial_latent) / float(city_properties['武汉市']['pop'])  # 1e-4
        # Rs0[nodes['武汉市']] = float(first_cases) / float(city_properties['武汉市']['pop'])
        # Ss0[nodes['武汉市']] = 1 - Is0[nodes['武汉市']] - Rs0[nodes['武汉市']]

        result = odeint(diff, np.r_[Is0, Rs0, Ss0], timespan, args=(r0,  t_l, gamma, matrix))

        sz = result.shape[1] // 3
        Is = result[:, :sz]  # infected 感染
        Rs = result[:, sz:2 * sz]  # recover 康复
        Ss = result[:, 2 * sz:3 * sz]  # susceptible 易感

        # 选取其中50day-150day 的数据 因为这部分数据会有趋势 500 个
        # Is_data = Is[250:, :]
        # Rs_data = Rs[250:, :]
        # Ss_data = Ss[250:, :]
        #选取其中50day-200day 的数据
        Is = Is[167:667, :]
        Rs = Rs[167:667, :]
        Ss = Ss[167:667, :]

        len_node = Is.shape[1]
        Is_list = np.zeros((500 // args.sample_freq, len_node))
        Rs_list = np.zeros((500 // args.sample_freq, len_node))
        Ss_list = np.zeros((500 // args.sample_freq, len_node))
        for j in range(Is_list.shape[0]):
            Is_list[j] = Is[args.sample_freq * j, :]
            Rs_list[j] = Rs[args.sample_freq * j, :]
            Ss_list[j] = Ss[args.sample_freq * j, :]

        Is_data2 = Is_list[:, :, np.newaxis]
        Rs_data2 = Rs_list[:, :, np.newaxis]
        Ss_data2 = Ss_list[:, :, np.newaxis]
        simu = np.concatenate((Is_data2, Rs_data2, Ss_data2), axis=2)

        if(i==0):
            simu_all = simu
        else :
            simu_all =  np.concatenate((simu_all,simu),axis=0)

    print(simu_all.shape)

    results = [matrix, simu_all]
    data_path = '/data/zhangyan/SIR/new_SIR_weight_times_'+str(10* args.times)+'.pickle'

    with open(data_path, 'wb') as f:
        pickle.dump(results, f)


#生成一个图
edges= generate_network()
print('开始生成数据')
start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('start_time:', start_time)
generate_data(edges)
print('数据生成完成')
end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
print('end_time:', end_time)
#print('time for generate data:' + str(round(end_time - start_time, 2)))