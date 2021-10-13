import matplotlib.style
import matplotlib as mpl
mpl.style.use('classic')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import os
from itertools import combinations
import sympy as sp
from sympy import *
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
WHETHER_TRAIN = False


varia_dict = {
    'age':'t',
    'MAT': 'T',
    'precipitation':'P',
    'erosion':'E',
    'quartz':'Q',
    'albite':'A'
}

### PARAMETEER LIST
# iteration_symbol = 'Y'
each_variables = ['age','MAT','precipitation','erosion','quartz','albite']
# for a in each_variables:

res = [list(com) for sub in range(len(each_variables)) for com in combinations(each_variables, sub + 1)]

total_summary = {
    'iteration': [],  ##  A. Hybrid model fit to SCT1,2,3,5
    'function': [],  ##  f(t,a,Q)
    'Targetted Soil(s)': [],  # Panola
    'Hybrid Model MSE': [],
    'Physics-based Model MSE': [],
    'f(z) (hybrid model)': [],
    'gk (physics)': [],
}
####

NUM_STEPS = 15000
lr = 0.001
training_locs_list = [
['SCT1', 'SCT2', 'SCT3', 'SCT5','Panola'],
['SCT1', 'SCT2', 'SCT3', 'SCT5','Davis'],
['SCT1', 'SCT2', 'SCT3', 'SCT5']
]


for variables, iteration_sym in zip(res,range(len(res))):
    iteration_sym +=1
    NUM_VARIABLE = len(variables)
    suffix_list = "abc"
    for training_locs, suffix in zip(training_locs_list,suffix_list):
        if suffix == 'c' and 'MAT' in variables:
            continue
        if suffix == 'c' and 'precipitation' in variables:
            continue
        if suffix == 'c' and 'erosion' in variables:
            continue




        tf.reset_default_graph()

        iteration_symbol = str(iteration_sym)
        iteration_symbol += suffix


        # training_locs = ['SCT1', 'SCT2', 'SCT3', 'SCT5','Panola']
        # testing_locs = ['panola','davis','JH5']
        location_list = ['SCT1','SCT2','SCT3','SCT5','Panola','Davis','Jughandle']
        testing_locs = [loc for loc in location_list if loc not in training_locs ]
        # ['sct1','sct2','sct3','sct5']
        # ['sct1' ,'sct2','sct3','sct5','davis']
        HOME_DIR = "/Users/chenchacha/sue_project/results"
        PATH = os.path.join(HOME_DIR,'iter_{0}'.format(iteration_symbol))
        if not os.path.exists(PATH):
            os.mkdir(PATH)


        def cal_A(c0,cx0):
            c0 = np.array(c0)
            cx0 = np.array(cx0)
            A = (c0 - cx0) / cx0
            return A

        ## data initialiåtion
        data = pd.read_csv('/Users/chenchacha/sue_project/data/all_data-new.csv')
        # variable_list = ['age','MAT','precipitation','erosion','quartz','albite'] ##t,T,P,E,q,a


        def formula(depth,A,gk):
            # depth = input_layer[:, 0:1]
            # c0 = input_layer[:, 1:2]
            # gk = input_layer[:, 2:3]
            # cx0 = input_layer[:, 3:4]
            # A = (c0 - cx0) / cx0
            return 1 / (1 + A * tf.math.exp(gk * depth))

        def tt_module(Z,hsize=[16, 8]):
            ### set 0-3
            # h1 = tf.layers.dense(Z,hsize[0])
            h2 = tf.layers.dense(Z, 16, activation='sigmoid')
            # out = tf.layers.dense(h2,8,activation='sigmoid')
            out = tf.layers.dense(h2,1,activation='linear')
            return out

        def analytical(depth, A, tt_output):
            return 1 / (1 + A * tf.math.exp(tt_output * depth))


        depth = tf.placeholder(tf.float32,[None,1]) ## depth
        Z = tf.placeholder(tf.float32,[None,NUM_VARIABLE])  ## Age, MAT, precipitation,erosion,QUARTZ
        A = tf.placeholder(tf.float32,[None,1])  ##
        concentration = tf.placeholder(tf.float32,[None,1])
        gk = tf.placeholder(tf.float32,[None,1])

        tt_output = tt_module(Z)
        concentration_pred = analytical(depth,A,tt_output)
        phy_pred = formula(depth,A,gk)
        loss = tf.reduce_mean(tf.squared_difference(concentration_pred, concentration))
        phy_loss = tf.reduce_mean(tf.squared_difference(phy_pred,concentration))
        step = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss) # G Train step


        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)
        saver = tf.train.Saver()


        data_tmp = data[data['location'].isin(training_locs)]

        depth_batch = np.array(data_tmp['depth']).reshape(data_tmp['depth'].shape[0],1)
        # Z_batch = np.array(data[['age','MAT']]).reshape(data['age'].shape[0],2)
        Z_batch = np.array(data_tmp[variables]).reshape(data_tmp['age'].shape[0],NUM_VARIABLE)
        gk_batch = np.array(data_tmp['gk']).reshape(data_tmp['gk'].shape[0],1)
        c0 = data_tmp[['c0']]
        cx0 = data_tmp[['cx0']]
        A_batch = cal_A(c0,cx0)
        concentration_batch = data_tmp[['concentration']]
        concentration_batch_normalize = np.array(concentration_batch)/np.array(c0)

        ## TRAINING

        print('==================== TRAINING ================================')
        if WHETHER_TRAIN:
            for i in range(NUM_STEPS):
                _, step_loss = sess.run([step,loss], feed_dict={depth: depth_batch, Z: Z_batch, A: A_batch, concentration: concentration_batch_normalize})
                if i%1000 == 0:
                    print(i,step_loss)
            save_path = saver.save(sess, os.path.join(PATH,"model.ckpt".format(iteration_symbol) ))
                                   # "model/model_{0}.ckpt".fo/rmat(iteration_symbol))
            print("Model saved in path: %s" % save_path)
            phy_loss_tmp = sess.run([phy_loss], feed_dict={depth: depth_batch, gk:gk_batch, A: A_batch, concentration: concentration_batch_normalize})
            print(phy_loss_tmp)

            loc_names = ''
            for i in training_locs:
                loc_names += (i+' ')
            loc_names = loc_names[:-1]
            function_name = 'f('
            # print(variables)
            for i in variables:
                function_name += varia_dict[i] + ','
            function_name = function_name[:-1]+')'
            total_summary['iteration'].append(str(iteration_symbol)+'. '+ function_name +' fit to '+ loc_names + '.')
            total_summary['function'].append(function_name)
            total_summary['Targetted Soil(s)'].append(loc_names)
            total_summary['Hybrid Model MSE'].append(step_loss)
            total_summary['Physics-based Model MSE'].append(phy_loss_tmp)
            total_summary['f(z) (hybrid model)'].append('-')
            total_summary['gk (physics)'].append('-')

            print("======== GENEEATE FUNCTION")
            input_mat = []
            for i in variables:
                input_mat.append(Symbol(varia_dict[i]))
            # input_mat = np.array([age, temperature, precipitation, quartz])
            input_mat = np.array(input_mat)

            for i in range(int(len(tf.trainable_variables()) / 2)):

                w = tf.trainable_variables()[2 * i].eval(sess)
                b = tf.trainable_variables()[2 * i + 1].eval(sess)
                input_mat = np.dot(input_mat, w) + b
                if i == 0:
                    input_mat = [1 / (sp.exp(-a) + 1) for a in input_mat]
                # if i == 1:
                #     # input_mat = [sp.tanh(a) for a in input_mat]
                #     input_mat = [2 / (sp.exp(-2*a) + 1) -1 for a in input_mat]

            # print(input_mat)
            assert len(input_mat)==1
            text_file = open(os.path.join(PATH,'Write_out_function.txt'), "w")
            text_file.write("%s" % str(input_mat[0]))
            text_file.close()

        ## TESTING
        print('==================== TESTING AND PLOTTING =================================')
        saver.restore(sess, os.path.join(PATH,"model.ckpt".format(iteration_symbol)))

        for loc in testing_locs:
            print("testing on {0}".format(loc))
            data_tmp = data[data['location']==loc]
            depth_batch_test = np.array(data_tmp['depth']).reshape(data_tmp['depth'].shape[0],1)


            # Z_batch = np.array(data[['age','MAT']]).reshape(data['age'].shape[0],2)
            Z_batch_test = np.array(data_tmp[variables]).reshape(data_tmp['age'].shape[0], NUM_VARIABLE)

            gk_batch_test = np.array(data_tmp['gk']).reshape(data_tmp['gk'].shape[0], 1)
            c0 = data_tmp[['c0']]
            cx0 = data_tmp[['cx0']]
            A_batch_test = cal_A(c0, cx0)
            concentration_batch_test = data_tmp[['concentration']]
            concentration_batch_normalize_test = np.array(concentration_batch_test) / np.array(c0)

            test_loss,phy_loss_tmp, phy_pred_tmp,concentration_pred_tmp, concentration_measured,gk_pred =sess.run([loss,phy_loss,phy_pred,concentration_pred,concentration,tt_output],
                              feed_dict = {depth: depth_batch_test, gk:gk_batch_test, Z: Z_batch_test, A:A_batch_test, concentration: concentration_batch_normalize_test})

            # print("hybrid  loss", test_loss)
            # print("physics loss", phy_loss_tmp)
            # print("hybrid  gamma*k",gk_pred[0][0])
            # print("physics  gamma*k",gk_batch_test[0][0])
            # loc_names = loc
            loc_names = loc
            function_name = 'f('
            # print(variables)

            for i in variables:
                function_name += varia_dict[i] + ','
            function_name = function_name[:-1]+')'
            total_summary['iteration'].append(str(iteration_symbol) + '. ' + function_name + ' test on ' + loc_names + '.')
            total_summary['function'].append(function_name)
            total_summary['Targetted Soil(s)'].append(loc_names)

            total_summary['Hybrid Model MSE'].append(test_loss)
            total_summary['Physics-based Model MSE'].append(phy_loss_tmp)
            total_summary['f(z) (hybrid model)'].append(gk_pred[0][0])
            total_summary['gk (physics)'].append(gk_batch_test[0][0])

            ## draw training figure
            if loc == 'Davis':
                times = 30
            elif loc =='Panola':
                times = 23
            else:
                times = 10  ## JH5

            a_new = np.repeat(A_batch_test[0], times).reshape(-1, 1)
            # A_batch_test
            atmp = np.repeat([Z_batch_test[0]], times,axis=0).reshape(-1, NUM_VARIABLE)
            assert np.array(atmp[0] == Z_batch_test[0]).all()
            # btmp = np.repeat(Z_batch_test_t_T[0][1],times)
            # z_new = np.column_stack((atmp,btmp)).reshape(-1,2)
            gk_batch_new = np.repeat(gk_batch_test[0], times).reshape(-1, 1)

            depth_batch_test_new = np.array(range(-times, 0, 1)).reshape(-1, 1)
            concentration_pred_tmp, phy_pred_tmp = sess.run([concentration_pred, phy_pred],
                                                                feed_dict={depth: depth_batch_test_new, Z: atmp, A: a_new, gk: gk_batch_new})

            # plt.scatter(np.array(depth_batch_test).reshape(-1,1), np.array(concentration).reshape(-1,1),label='Real')
            # plt.plot(np.array(depth_batch_test).reshape(-1,1), np.array(concentration).reshape(-1,1))
            plt.figure()
            plt.scatter(np.array(concentration_measured).reshape(-1, 1), np.array(depth_batch_test).reshape(-1, 1),
                        label='Measured', s=150, edgecolors='b')

            # plt.plot( np.array(concentration).reshape(-1,1), np.array(depth_batch_test).reshape(-1,1))


            # plt.scatter(np.array(concentration_pred).reshape(-1,1), np.array(depth_batch_test).reshape(-1,1), label='Hybrid Model',s=30)
            plt.plot(np.array(concentration_pred_tmp).reshape(-1, 1), np.array(depth_batch_test_new).reshape(-1, 1),
                     label='Hybrid Model', linewidth=4, alpha=0.7, c='r')

            # plt.scatter(np.array(phy_pred).reshape(-1,1), np.array(depth_batch_test).reshape(-1,1), label='Physics-based Model',s=30)
            plt.plot(np.array(phy_pred_tmp).reshape(-1, 1), np.array(depth_batch_test_new).reshape(-1, 1),
                     label='Physics-based Model', linewidth=4, alpha=0.7, c='green')
            # plt.title('Davis', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.xlim(0, 1.2)
            plt.xlabel('Concentration (mol Na m$^{-3}$ regolith)', fontsize=20)
            plt.ylabel('Depth (m)', fontsize=20)
            plt.legend(fontsize=20)
            plt.tight_layout()
            plt.savefig(os.path.join(PATH, 'fig_{0}_{1}.pdf'.format(loc, iteration_symbol)))
            plt.clf()
            plt.close()

# print(total_summary)
total_summary = pd.DataFrame(total_summary)

# total_summary.to_csv(os.path.join(HOME_DIR,'total_results_davis_20.csv'),sep=';',index=False)