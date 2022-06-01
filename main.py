"""
Created on Thu May 27 2022

@author: xiaoang zhang, leon mayer
"""
import numpy as np
from CMAC import CMAC
import pickle
#from matplotlib import pyplot as plt

def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def prepare_cmac():
    #load .npy file
    training_data = np.load('/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/samples_straighthand.npy')
    x = training_data[:, :2]
    y = training_data[:, 2:]
    max_y = np.amax(y, axis=0)
    min_y = np.amin(y, axis=0)
    n_input = 2
    n_output = 2
    n_a = 5

    res = np.array([50, 50])
    disp3 = np.array([[0, 2], [1, 1], [2, 0]])
    disp5 = np.array([[3, 2], [4, 2], [1, 1], [3, 0], [0, 4]])
    cmac = CMAC(n_input, n_output, n_a, res, disp5, max_y, min_y)
    return cmac, x, y

                
if __name__ == '__main__':
    
    epochs = 10

    num_samples = [150]
    samples_list = []
    for k in num_samples:
        cmac, x_target, y = prepare_cmac()
        y_selected = y[0:k, :]
        x_target_selected = x_target[0:k, :]
        
        mse_list = []
        for j in range(epochs):
            x_output_list = []
            for i in range(0, y_selected.shape[0]):
                x = cmac.cmacMap(y_selected[i,:])
                x_output_list.append(x)
                W = cmac.cmacTargetTrain(x_target_selected[i,:], x)
            x_array = np.asarray(x_output_list)
            mse_list.append((((x_array - x_target_selected) ** 2).mean()))
        samples_list.append(mse_list)
<<<<<<< HEAD
        np.save('/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/W_straighthand.npy', cmac.W)
        #save_object(cmac, '/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/trained_cmac.pkl')
=======
        np.save('/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/W.npy', cmac.W)
        save_object(cmac, '/home/bio/ros/bioinspired_ws/src/tutorial_3/scripts/trained_cmac.pkl')
>>>>>>> ac758ec3605f9553eb6a062b45f4ab1679deb6df
    '''
    for sample, label1 in zip(samples_list, num_samples):
        plt.plot(sample, label=(str(label1) + " samples"))
    plt.legend(loc="upper right")
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.title('MSE for 75 and 150 training samples')
    plt.savefig("C:/software/Studium/Master/bilhr/MSEboth_rec5.png")
    plt.show()
    '''