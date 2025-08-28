## Created by Ayan Banerjee, Arizona State University
## Property of IMPACT Lab ASU

import numpy as np
import pandas as pd
import csv
import os
import pdb
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'  # expose GPU 0

import tensorflow.compat.v1 as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse

numID = 1

def cut_in_sequences(x,y,seq_len,inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0,x.shape[0] - seq_len,inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x,axis=1),np.stack(sequences_y,axis=1)

#@tf.function(input_signature=[tf.TensorSpec(shape=None,dtype=tf.float32)])
#def compare(x):
#    a = np.float32(np.array([10000.0]))
#    stableEps = tf.constant(a)
#    result = tf.math.greater(x,stableEps)
#    return result

Nloop = 0;


class Custom_CE_Loss(tf.keras.losses.Loss):
    def __init__(self,labels,logits,uMotor1,uMotor2,uMotor3,uMotor4,maxMotor,minMotor):
        self.y_true2 = labels
        self.y_pred2 = logits
        self.y_uMotor1 = uMotor1
        self.y_uMotor2 = uMotor2
        self.y_uMotor3 = uMotor3
        self.y_uMotor4 = uMotor4
        self.y_maxMotor = maxMotor
        self.y_minMotor = minMotor
        
        

    
        
    def lossF(self):        
        ############################# Parameters ##################################
        maxChange = 95;
        dxm = (1+(0.5-self.y_pred2[:, :, 0]) * maxChange / 100) * 0.16 # arm length (m)
        dym = (1 + (0.5 - self.y_pred2[:, :, 1]) * maxChange / 100) * 0.16 # arm length (m)

            

        mB = 1.2 # kg
        g = 9.81 # m/s/s
        dzm = (1 + (0.5 - self.y_pred2[:, :, 2]) * maxChange / 100) * 0.05 # motor height (m)
        IBxx = (1 + (0.5 - self.y_pred2[:, :, 3]) * maxChange / 100) * 0.0123 # inertia
            
            
        IByy = (1 + (0.5 - self.y_pred2[:, :, 4]) * maxChange / 100) * 0.0123 # inertia
            
        IBzz = (1 + (0.5 - self.y_pred2[:, :, 5]) * maxChange / 100) * 0.0123 # inertia
        Cd = (1 + (0.5 - self.y_pred2[:, :, 6]) * maxChange / 100) * 0.1

            
        kTh = (1 + (0.5 - self.y_pred2[:, :, 7]) * maxChange / 100) * 1.076e-5 # thrust coeff (N/(rad/s)/(rad/s)) (1.18 e -7 N/RPM/RPM)
        kTo = (1 + (0.5 - self.y_pred2[:, :, 8]) * maxChange / 100) * 1.632e-7 # thrust coeff (Nm/(rad/s)/(rad/s)) (1.79 e -9 Nm/RPM/RPM)
    
            
        tau2 = (1 + (0.5 - self.y_pred2[:, :, 9]) * maxChange / 100) * 0.015 # Value for second order system for Motor dynamics

            

        kp = (1 + (0.5 - self.y_pred2[:, :, 10]) * maxChange / 100) * 1 # Value for second order system for Motor dynamics
        damp = (1 + (0.5 - self.y_pred2[:, :, 11]) * maxChange / 100) * 1 # Value for second order system for Motor dynamics


        


        ###########################################################################
        #print("true2 shape: {}".format(self.y_true2))
         
        x_z = self.y_true2[:,:,0]
        x_x = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        x_y = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        
        #print("Gval shape is {}".format(GVal))
        quat0 = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        quat1 = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        quat2 = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        quat3 = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        
        xdot = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        ydot = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        zdot = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        
        p = 0
        q = 0
        r = 0
        
        w_hover1 = tf.math.divide_no_nan(tf.math.multiply(mB,g)/4,kTh)
        wdot_hover1 = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        w_hover2 = tf.math.divide_no_nan(tf.math.multiply(mB,g)/4,kTh)
        wdot_hover2 = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        w_hover3 = tf.math.divide_no_nan(tf.math.multiply(mB,g)/4,kTh)
        wdot_hover3 = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)
        w_hover4 = tf.math.divide_no_nan(tf.math.multiply(mB,g)/4,kTh)
        wdot_hover4 = tf.zeros(tf.shape(x_z),dtype=tf.dtypes.float32)        
        
                        
        x_x = tf.expand_dims(x_x,2)
        x_y = tf.expand_dims(x_y,2)
        x_z = tf.expand_dims(x_z,2)
        quat0 = tf.expand_dims(quat0,2)
        quat1 = tf.expand_dims(quat1,2)
        quat2 = tf.expand_dims(quat2,2)
        quat3 = tf.expand_dims(quat3, 2)
        xdot = tf.expand_dims(xdot, 2)
        ydot = tf.expand_dims(ydot,2)
        zdot = tf.expand_dims(zdot,2)
        p = tf.expand_dims(p,2)
        q = tf.expand_dims(q, 2)
        r = tf.expand_dims(r, 2)
        w_hover1 = tf.expand_dims(w_hover1, 2)
        w_hover2 = tf.expand_dims(w_hover2, 2)
        w_hover3 = tf.expand_dims(w_hover3, 2)
        w_hover4 = tf.expand_dims(w_hover4, 2)
        
        wdot_hover1 = tf.expand_dims(wdot_hover1, 2)
        wdot_hover2 = tf.expand_dims(wdot_hover2, 2)
        wdot_hover3 = tf.expand_dims(wdot_hover3, 2)
        wdot_hover4 = tf.expand_dims(wdot_hover4, 2)
        
        
        
        limitLoop = 144
        tau = 0.00005
        a = np.float32(np.array([100000.0]))
        stableEps = tf.constant(a)
        
        

        
        eps = 25
        for i in range(1,limitLoop,1):
            
            
            dummy_x_x = x_x[:, :, i - 1] + tau * xdot[:, :, i-1]
            dummy_x_y = x_y[:, :, i-1] + tau * ydot[:, :, i-1]
            dummy_x_z = x_z[:, :, i-1] + tau * zdot[:, :, i-1]
            dummyq0 = quat0[:, :, i-1] + tau * (-0.5*tf.math.multiply(p[:,:,i-1],quat1[:,:,i-1])-0.5*tf.math.multiply(q[:,:,i-1],quat2[:,:,i-1])-0.5*tf.math.multiply(r[:,:,i-1],quat3[:,:,i-1]))
            
            dummyq1 = quat1[:, :, i-1] + tau * (0.5*tf.math.multiply(p[:,:,i-1],quat0[:,:,i-1])-0.5*tf.math.multiply(q[:,:,i-1],quat3[:,:,i-1])+0.5*tf.math.multiply(r[:,:,i-1],quat2[:,:,i-1]))
            
            dummyq2 = quat2[:, :, i-1] + tau * (0.5*tf.math.multiply(p[:,:,i-1],quat3[:,:,i-1])+0.5*tf.math.multiply(q[:,:,i-1],quat0[:,:,i-1])-0.5*tf.math.multiply(r[:,:,i-1],quat1[:,:,i-1]))
            
            dummyq3 = quat3[:, :, i-1] + tau * (-0.5*tf.math.multiply(p[:,:,i-1],quat2[:,:,i-1])+0.5*tf.math.multiply(q[:,:,i-1],quat1[:,:,i-1])+0.5*tf.math.multiply(r[:,:,i-1],quat0[:,:,i-1]))
            
            
            wddotM1 = (-2.0* tf.math.multiply(tf.math.multiply(damp,tau2),wdot_hover1[:, :, i-1])-w_hover1[:, :, i-1] + tf.math.divide_no_nan(tf.math.multiply(kp,self.y_uMotor1[:, :, i-1]),tf.math.square(tau2))
            wddotM2 = (-2.0* tf.math.multiply(tf.math.multiply(damp,tau2),wdot_hover2[:, :, i-1])-w_hover2[:, :, i-1] + tf.math.divide_no_nan(tf.math.multiply(kp,self.y_uMotor2[:, :, i-1]),tf.math.square(tau2))
            wddotM3 = (-2.0* tf.math.multiply(tf.math.multiply(damp,tau2),wdot_hover1[:, :, i-1])-w_hover1[:, :, i-1] + tf.math.divide_no_nan(tf.math.multiply(kp,self.y_uMotor3[:, :, i-1]),tf.math.square(tau2))
            wddotM4 = (-2.0* tf.math.multiply(tf.math.multiply(damp,tau2),wdot_hover1[:, :, i-1])-w_hover1[:, :, i-1] + tf.math.divide_no_nan(tf.math.multiply(kp,self.y_uMotor4[:, :, i-1]),tf.math.square(tau2))
            
            w_hover1[:, :, i-1] = tf.clip_by_value(w_hover1[:, :, i-1], self.y_minMotor, self.y_maxMotor)
            w_hover2[:, :, i-1] = tf.clip_by_value(w_hover2[:, :, i-1], self.y_minMotor, self.y_maxMotor)
            w_hover3[:, :, i-1] = tf.clip_by_value(w_hover3[:, :, i-1], self.y_minMotor, self.y_maxMotor)
            w_hover4[:, :, i-1] = tf.clip_by_value(w_hover4[:, :, i-1], self.y_minMotor, self.y_maxMotor)
            
            
            ThrM1 = tf.math.multiply(kTh,tf.math.square(w_hover1[:,:,i-1]))
            ThrM2 = tf.math.multiply(kTh,tf.math.square(w_hover2[:,:,i-1]))
            ThrM3 = tf.math.multiply(kTh,tf.math.square(w_hover3[:,:,i-1]))
            ThrM4 = tf.math.multiply(kTh,tf.math.square(w_hover4[:,:,i-1]))
            TorM1 = tf.math.multiply(kTo,tf.math.square(w_hover1[:,:,i-1]))
            TorM2 = tf.math.multiply(kTo,tf.math.square(w_hover2[:,:,i-1]))
            TorM3 = tf.math.multiply(kTo,tf.math.square(w_hover3[:,:,i-1]))
            TorM4 = tf.math.multiply(kTo,tf.math.square(w_hover4[:,:,i-1]))            
            
            # wind effect not modeled
            
            dummyxdot = xdot[:, :, i-1] + tau * (tf.math.multiply(tf.math.multiply(Cd,tf.math.sign(-xdot[:,:,i-1])),tf.math.square(xdot[:,:,i-1])) + 2*tf.math.multiply((tf.math.multiply(quat0,quat2)+tf.math.multiply(quat1,quat3)),(ThrM1+ThrM2+ThrM3+ThrM4)))/mB
            dummyydot = ydot[:, :, i-1] + tau * (tf.math.multiply(tf.math.multiply(Cd,tf.math.sign(-ydot[:,:,i-1])),tf.math.square(ydot[:,:,i-1])) - 2*tf.math.multiply((tf.math.multiply(quat0,quat1)-tf.math.multiply(quat2,quat3)),(ThrM1+ThrM2+ThrM3+ThrM4)))/mB            
            dummyzdot = zdot[:, :, i-1] + tau * (tf.math.multiply(tf.math.multiply(-Cd,tf.math.sign(zdot[:,:,i-1])),tf.math.square(zdot[:,:,i-1])) + tf.math.multiply((ThrM1+ThrM2+ThrM3+ThrM4),(tf.math.square(quat0[:,:,i-1])-tf.math.square(quat1[:,:,i-1])-tf.math.square(quat2[:,:,i-1])+tf.math.square(quat3[:,:,i-1])))-g*mB)/mB    
            
            
            dummyp = p[:,:,i-1] + tau * tf.math.divide_no_nan((tf.math.multiply(tf.math.multiply(IByy-IBzz,q[:,:,i-1]),r[:,:,i-1]) + tf.math.multiply(tf.math.multiply(IRzz,(w_hover1[:,:,i-1]-w_hover2[:,:,i-1]+w_hover3[:,:,i-1]-w_hover4[:,:,i-1])),q[:,:,i-1])+tf.math.multiply((ThrM1-ThrM2-ThrM3+ThrM4),dym)),IBxx)
            dummyq = q[:,:,i-1] + tau * tf.math.divide_no_nan((tf.math.multiply(tf.math.multiply(IBzz-IBxx,p[:,:,i-1]),r[:,:,i-1]) - tf.math.multiply(tf.math.multiply(IRzz,(w_hover1[:,:,i-1]-w_hover2[:,:,i-1]+w_hover3[:,:,i-1]-w_hover4[:,:,i-1])),p[:,:,i-1])+tf.math.multiply((-ThrM1-ThrM2+ThrM3+ThrM4),dxm)),IByy)
            dummyr = r[:,:,i-1] + tau * tf.math.divide_no_nan((tf.math.multiply(tf.math.multiply(IBxx-IBzz,p[:,:,i-1]),q[:,:,i-1]) + TorM1-TorM2+TorM3-TorM4),IBzz)
            
            
            dummy_w_hover1 = w_hover1[:, :, i-1] + tau*wdot_hover1[:,:,i-1]
            dummy_w_hover2 = w_hover2[:, :, i-1] + tau*wdot_hover2[:,:,i-1]
            dummy_w_hover3 = w_hover3[:, :, i-1] + tau*wdot_hover3[:,:,i-1]
            dummy_w_hover4 = w_hover4[:, :, i-1] + tau*wdot_hover4[:,:,i-1] 
            
            dummy_wdot_hover1 = wdot_hover1[:,:,i-1] + tau*wddotM1
            dummy_wdot_hover2 = wdot_hover2[:,:,i-1] + tau*wddotM2
            dummy_wdot_hover3 = wdot_hover3[:,:,i-1] + tau*wddotM3
            dummy_wdot_hover4 = wdot_hover4[:,:,i-1] + tau*wddotM4
            
            
            
                    
                        
            
            
            max_inc = 400.0  # max mg/dL per timestep you consider safe
            mask    = tf.logical_and(
                          tf.math.is_finite(dummy_x_z),
                          tf.abs(dummy_x_z - x_z[:,:,i-1]) <= max_inc
                      )

            # 3) select either the new value or carry forward the old
            dummy_x_z = tf.where(mask,
                          dummy_x_z,   # if safe
                          x_z[:,:,i-1]    # otherwise no change
                         )

                      
             
            
            
            
            dummy_x_x = tf.expand_dims(dummy_x_x,2)
            x_x = tf.concat([x_x,dummy_x_x],2)
            dummy_x_y = tf.expand_dims(dummy_x_y, 2)
            x_y = tf.concat([x_y, dummy_x_y], 2)
            dummy_x_z = tf.expand_dims(dummy_x_z, 2)
            x_z = tf.concat([x_z, dummy_x_z], 2)
            dummyq0 = tf.expand_dims(dummyq0, 2)
            quat0 = tf.concat([quat0, dummyq0], 2)
            dummyq1 = tf.expand_dims(dummyq1, 2)
            quat1 = tf.concat([quat1, dummyq1], 2)
            dummyq2  = tf.expand_dims(dummyq2, 2)
            quat2 = tf.concat([quat2, dummyq2], 2)
            dummyq3 = tf.expand_dims(dummyq3,2)
            quat3 = tf.concat([quat3, dummyq3],2)
            dummyxdot = tf.expand_dims(dummyxdot, 2)
            xdot = tf.concat([xdot, dummyxdot], 2)
            dummyydot = tf.expand_dims(dummyydot, 2)
            ydot = tf.concat([ydot, dummyydot], 2)
            dummyzdot = tf.expand_dims(dummyzdot, 2)
            zdot = tf.concat([zdot, dummyzdot], 2)
            dummyp = tf.expand_dims(dummyp, 2)
            p = tf.concat([p, dummyp], 2)
            dummyq = tf.expand_dims(dummyq, 2)
            q = tf.concat([q, dummyq], 2)
            dummyr = tf.expand_dims(dummyr, 2)
            r = tf.concat([r, dummyr], 2)
            dummy_w_hover1 = tf.expand_dims(dummy_w_hover1, 2)
            w_hover1 = tf.concat([w_hover1, dummy_w_hover1], 2)
            dummy_w_hover2 = tf.expand_dims(dummy_w_hover2, 2)
            w_hover2 = tf.concat([w_hover2, dummy_w_hover2], 2)
            dummy_w_hover3 = tf.expand_dims(dummy_w_hover3, 2)
            w_hover3 = tf.concat([w_hover3, dummy_w_hover3], 2)
            dummy_w_hover4 = tf.expand_dims(dummy_w_hover4, 2)
            w_hover4 = tf.concat([w_hover4, dummy_w_hover4], 2)
            dummy_wdot_hover1 = tf.expand_dims(dummy_wdot_hover1, 2)
            wdot_hover1 = tf.concat([wdot_hover1, dummy_wdot_hover1], 2) 
            dummy_wdot_hover2 = tf.expand_dims(dummy_wdot_hover2, 2)
            wdot_hover2 = tf.concat([wdot_hover2, dummy_wdot_hover2], 2)
            dummy_wdot_hover3 = tf.expand_dims(dummy_wdot_hover3, 2)
            wdot_hover3 = tf.concat([wdot_hover3, dummy_wdot_hover3], 2)
            dummy_wdot_hover4 = tf.expand_dims(dummy_wdot_hover4, 2)
            wdot_hover4 = tf.concat([wdot_hover4, dummy_wdot_hover4], 2)           
            #print("Isc1Val: {}".format(Isc1Val))
            #print("kd: {}".format(kd))
            #print("self.y_basal: {}".format(self.y_basal))
            #print("self.y_ins: {}".format(self.y_ins)) 
            #print("Vi: {}".format(Vi)) 


        err = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None])
        
        
        
        return err

    

    

        

class HarData:

    def __init__(self,seq_len=16):
        tf.compat.v1.disable_eager_execution()
        tf.disable_v2_behavior()
        print("Parsing for Patient File {}".format(numID))
        train_x_x = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/xData.txt")
        train_x_y = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/yData.txt")
        train_x_z = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/zData.txt")
        
        train_y_x = (np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/xData.txt")-1)#.astype(np.int32)
        train_y_y = (np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/yData.txt")-1)#.astype(np.int32)
        train_y_z = (np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/zData.txt")-1)#.astype(np.int32)
        
        train_uMotor1 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/uMotor1.txt")
        train_uMotor2 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/uMotor2.txt")
        train_uMotor3 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/uMotor3.txt")
        train_uMotor4 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/uMotor4.txt")
        
        train_maxMotor = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/maxMotor.txt")
        train_minMotor = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/minMotor.txt")
                
        
        
        
        test_x_x = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/xData.txt")
        test_x_y = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/yData.txt")
        test_x_z = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/zData.txt")
        
        test_y_x = (np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/xData.txt")-1)#.astype(np.int32)
        test_y_y = (np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/yData.txt")-1)#.astype(np.int32)
        test_y_z = (np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/zData.txt")-1)#.astype(np.int32)
        
        test_uMotor1 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/uMotor1.txt")
        test_uMotor2 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/uMotor2.txt")
        test_uMotor3 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/uMotor3.txt")
        test_uMotor4 = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/uMotor4.txt")
        
        test_maxMotor = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/maxMotor.txt")
        test_minMotor = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/minMotor.txt")
        

        
        shape = tf.shape(test_basal)
        with tf.compat.v1.Session() as sess:
            numpy_number = sess.run(shape[1])
        Nloop = numpy_number
        print("Nloop {}".format(Nloop))
        train_x_x,train_y_x = cut_in_sequences(train_x_x,train_y_x,seq_len)
        train_x_y,train_y_y = cut_in_sequences(train_x_y,train_y_y,seq_len)
        train_x_z,train_y_z = cut_in_sequences(train_x_z,train_y_z,seq_len)
        
        train_uMotor1,train_uMotor2 = cut_in_sequences(train_uMotor1,train_uMotor2,seq_len)
        train_uMotor3,train_uMotor4 = cut_in_sequences(train_uMotor3,train_uMotor4,seq_len)
        train_maxMotor, train_minMotor = cut_in_sequences(train_maxMotor, train_minMotor, seq_len)
        
        
        
        ## initiate Isc1 and Isc2
        
        
        ##
        

        test_x_x,test_y_x = cut_in_sequences(test_x_x,test_y_x,seq_len,inc=8)
        test_x_y,test_y_y = cut_in_sequences(test_x_y,test_y_y,seq_len,inc=8)
        test_x_z,test_y_z = cut_in_sequences(test_x_z,test_y_z,seq_len,inc=8)

        test_uMotor1,test_uMotor2 = cut_in_sequences(test_uMotor1,test_uMotor2,seq_len,inc=8)
        test_uMotor3,test_uMotor4 = cut_in_sequences(test_uMotor3,test_uMotor4,seq_len,inc=8)
        test_maxMotor, test_minMotor = cut_in_sequences(test_maxMotor, test_minMotor, seq_len,inc=8)        
        
        
        #print("Total number of testing sequences: {}".format(test_initIsc1.shape[1]))
        #permutation = np.random.RandomState(893429).permutation(train_x.shape[1])
        valid_size = int(0.1*train_x_x.shape[1])
        print("Validation split: {}, training split: {}".format(valid_size,train_x_x.shape[1]-valid_size))

        self.valid_x_x = train_x_x[:,:valid_size]
        self.valid_x_y = train_x_y[:,:valid_size]
        self.valid_x_z = train_x_z[:,:valid_size]
        self.valid_y_x = train_y_x[:,:valid_size]
        self.valid_y_y = train_y_y[:,:valid_size]
        self.valid_y_z = train_y_z[:,:valid_size]        
        
        self.valid_uMotor1 = train_uMotor1[:,:valid_size]
        self.valid_uMotor2 = train_uMotor2[:,:valid_size]
        self.valid_uMotor3 = train_uMotor3[:,:valid_size]
        self.valid_uMotor4 = train_uMotor4[:,:valid_size]        
        
        self.valid_maxMotor = train_maxMotor[:,:valid_size]
        self.valid_minMotor = train_minMotor[:,:valid_size] 
        
        
        
        
        
        self.train_x_x = train_x_x[:,valid_size:]
        self.train_x_y = train_x_y[:,valid_size:]
        self.train_x_z = train_x_z[:,valid_size:]
        self.train_y_x = train_y_x[:,valid_size:]
        self.train_y_y = train_y_y[:,valid_size:]
        self.train_y_z = train_y_z[:,valid_size:]        
        
        self.train_uMotor1 = train_uMotor1[:,valid_size:]
        self.train_uMotor2 = train_uMotor2[:,valid_size:]
        self.train_uMotor3 = train_uMotor3[:,valid_size:]
        self.train_uMotor4 = train_uMotor4[:,valid_size:]        
        
        self.train_maxMotor = train_maxMotor[:,valid_size:]
        self.train_minMotor = train_minMotor[:,valid_size:]
        
        
        
               

        self.test_x_x = test_x_x
        self.test_x_y = test_x_y
        self.test_x_z = test_x_z


        self.test_y_x = test_y_x
        self.test_y_y = test_y_y
        self.test_y_z = test_y_z
        
        self.test_uMotor1 = test_uMotor1
        self.test_uMotor2 = test_uMotor2        
        self.test_uMotor3 = test_uMotor3
        self.test_uMotor4 = test_uMotor4
        
        self.test_maxMotor = test_maxMotor
        self.test_minMotor = test_minMotor        
        
        
        
        

        #pdb.set_trace()

        print("Total number of test sequences: {}".format(self.test_x_x.shape[1]))

    def iterate_train(self,batch_size=32):
        total_seqs = self.train_x_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x_x = self.train_x_x[:,start:end]
            batch_x_y = self.train_x_y[:,start:end]
            batch_x_z = self.train_x_z[:,start:end]
            
            batch_y_x = self.train_y_x[:,start:end]
            batch_y_y = self.train_y_y[:,start:end]
            batch_y_z = self.train_y_z[:,start:end]

            batch_uMotor1 = self.train_uMotor1[:,start:end]
            batch_uMotor2 = self.train_uMotor2[:,start:end]
            batch_uMotor3 = self.train_uMotor3[:,start:end] 
            batch_uMotor4 = self.train_uMotor4[:,start:end]   
            
            batch_maxMotor = self.train_maxMotor[:,start:end] 
            batch_minMotor = self.train_minMotor[:,start:end]        
            
                        
            



            yield (batch_x_x,batch_x_y,batch_x_z,batch_y_x,batch_y_y,batch_y_z,batch_uMotor1,batch_uMotor2,batch_uMotor3,batch_uMotor4,batch_maxMotor,batch_minMotor)

class HarModel:

    def __init__(self,model_type,model_size,learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = None

        self.x_x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.x_y = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.x_z = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.target_y_x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.target_y_y = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.target_y_z = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.uMotor1 = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.uMotor2 = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.uMotor3 = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.uMotor4 = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.maxMotor = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        self.minMotor = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,Nloop])
        
        
        
         
        
        
        

        self.model_size = model_size
        head = self.x_z



        print("Beginning ")

        if(model_type == "lstm"):
            self.fused_cell = tf.nn.rnn_cell.LSTMCell(model_size)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            learning_rate = 0.01 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op = self.wm.get_param_constrain_op()
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)

        else:
            raise ValueError("Unknown model type '{}'".format(model_type))


        self.y = tf.compat.v1.layers.Dense(12,activation='sigmoid')(head) # Dense layer output should be same as the number of model parameter
        print("logit shape: ")
        print(str(self.y.shape))
        print("self.y: ")
        print(self.y)
        
        lossVal = Custom_CE_Loss(labels = self.target_y_z,logits = self.y,uMotor1 = self.uMotor1,uMotor2=self.uMotor2,uMotor3=self.uMotor3,uMotor4=self.uMotor4,maxMotor=self.maxMotor,minMotor=self.minMotor).lossF()
        print("lossVal {}".format(lossVal))


        self.loss = tf.reduce_mean(Custom_CE_Loss(labels = self.target_y_z,logits = self.y,uMotor1 = self.uMotor1,uMotor2=self.uMotor2,uMotor3=self.uMotor3,uMotor4=self.uMotor4,maxMotor=self.maxMotor,minMotor=self.minMotor).lossF())
        
        self.lossV2 = tf.reduce_mean(Custom_CE_Loss(labels = self.target_y_z,logits = self.y,uMotor1 = self.uMotor1,uMotor2=self.uMotor2,uMotor3=self.uMotor3,uMotor4=self.uMotor4,maxMotor=self.maxMotor,minMotor=self.minMotor).lossF())
        
        


        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        
        
        #gvs = optimizer.compute_gradients(self.loss)
        # clip each gradient by global norm of 5.0
        #clipped_gvs = [(tf.clip_by_norm(g, 100.0), v) for g, v in gvs]
        # apply the clipped gradients
        #self.train_step = optimizer.apply_gradients(clipped_gvs)
        
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(1.0,tf.int64), tf.cast(1.0,tf.int64)), tf.float32))

        # configure GPU memory growth and device placement


        config = tf.compat.v1.ConfigProto()


        config.gpu_options.allow_growth = True


        config.allow_soft_placement = True


        config.log_device_placement = False


        self.sess = tf.compat.v1.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())

        self.result_file = os.path.join("results","har","{}_{}.csv".format(model_type,model_size))
        if(not os.path.exists("results/har")):
            os.makedirs("results/har")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","har","{}".format(model_type))
        if(not os.path.exists("tf_sessions/har")):
            os.makedirs("tf_sessions/har")

        self.saver = tf.train.Saver()


    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,gesture_data,epochs,verbose=True,log_period=50):

        best_valid_loss = 10000
        best_loss = 10000
        best_valid_stats = (0,0,0,0,0,0,0,0)
        self.save()
        count = 0
        count2 = 0
        count3 = 0
        for e in range(epochs):
            if(e%log_period == 0):
                print("x data {}".format({self.x_z: gesture_data.test_x_z}.keys()))
                print("y data {}".format({self.target_y_z:gesture_data.valid_y_z}.keys()))

                test_acc,test_loss,test_lossV2 = self.sess.run([self.accuracy,self.loss,self.lossV2],{self.x_x:gesture_data.test_x_x,self.x_y:gesture_data.test_x_y,self.x_z:gesture_data.test_x_z,self.target_y_x: gesture_data.test_y_x,self.target_y_y: gesture_data.test_y_y,self.target_y_z: gesture_data.test_y_z,self.uMotor1: gesture_data.test_uMotor1,self.uMotor2: gesture_data.test_uMotor2,self.uMotor3: gesture_data.test_uMotor3,self.uMotor4: gesture_data.test_uMotor4,self.maxMotor: gesture_data.test_maxMotor,self.minMotor: gesture_data.test_minMotor})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x_x:gesture_data.valid_x_x,self.x_y:gesture_data.valid_x_y,self.x_z:gesture_data.valid_x_z,self.target_y_x: gesture_data.valid_y_x,self.target_y_y: gesture_data.valid_y_y,self.target_y_z: gesture_data.valid_y_z, self.uMotor1: gesture_data.valid_uMotor1, self.uMotor2: gesture_data.valid_uMotor2, self.uMotor3: gesture_data.valid_uMotor3,self.uMotor4: gesture_data.valid_uMotor4,self.maxMotor: gesture_data.valid_maxMotor,self.minMotor: gesture_data.valid_minMotor})
                # Accuracy metric -> higher is better
                if(valid_loss < best_valid_loss and e > 0):
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                            e,
                            np.mean(losses),np.mean(accs)*100,
                            valid_loss,valid_acc*100,
                            test_loss,test_lossV2,test_acc*100
                            )
                    self.save()

            losses = []
            mardLoss = []
            accs = []
            tY = []
            err = self.loss 
            print("err {}".format(err))
            #pdb.set_trace()
            #breakpoint()
            for batch_x_x,batch_x_y,batch_x_z,batch_y_x,batch_y_y,batch_y_z,batch_uMotor1,batch_uMotor2,batch_uMotor3,batch_uMotor4,batch_maxMotor,batch_minMotor in gesture_data.iterate_train(batch_size=16):
                acc,loss,lossV2,t_step,t_y = self.sess.run([self.accuracy,self.loss,self.lossV2,self.train_step,self.y],{self.x_x:batch_x_x,self.x_y:batch_x_y,self.x_z:batch_x_z,self.target_y_x: batch_y_x,self.target_y_y: batch_y_y,self.target_y_z: batch_y_z, self.uMotor1: batch_uMotor1, self.uMotor2: batch_uMotor2, self.uMotor3: batch_uMotor3,self.uMotor4: batch_uMotor4,self.maxMotor: batch_maxMotor,self.minMotor: batch_minMotor})
                
                print("loss iter: {} ".format(loss))
                #breakpoint()
                if(not self.constrain_op is None):
                    self.sess.run(self.constrain_op)
                tY.append(tf.reduce_mean(tf.reduce_mean(t_y,axis=0),axis=0))
                losses.append(loss)
                mardLoss.append(lossV2)
                accs.append(acc)
            losses2 = losses[:-1]    
            mardLosses2 = mardLoss[:-1]
            print("loss: {}".format(sum(losses2)/len(losses2)))

            newLoss = sum(losses2)/len(losses2)

            newMardLoss = sum(mardLosses2)/len(mardLosses2)


            tyMean = tf.reduce_mean(tf.reduce_mean(t_y,axis=0),axis=0)
            tyMean2 = self.sess.run(tyMean)
            maxChange = 95
            dxm = (1+(0.5-tyMean2[0]) * maxChange / 100) * 0.16 # arm length (m)
            dym = (1 + (0.5 - tyMean2[1]) * maxChange / 100) * 0.16 # arm length (m)

            

            mB = 1.2 # kg
            g = 9.81 # m/s/s
            dzm = (1 + (0.5 - tyMean2[2]) * maxChange / 100) * 0.05 # motor height (m)
            IBxx = (1 + (0.5 - tyMean2[3]) * maxChange / 100) * 0.0123 # inertia
            
            
            IByy = (1 + (0.5 - tyMean2[4]) * maxChange / 100) * 0.0123 # inertia
            
            IBzz = (1 + (0.5 - tyMean2[ 5]) * maxChange / 100) * 0.0123 # inertia
            Cd = (1 + (0.5 - tyMean2[ 6]) * maxChange / 100) * 0.1

            
            kTh = (1 + (0.5 - tyMean2[ 7]) * maxChange / 100) * 1.076e-5 # thrust coeff (N/(rad/s)/(rad/s)) (1.18 e -7 N/RPM/RPM)
            kTo = (1 + (0.5 - tyMean2[ 8]) * maxChange / 100) * 1.632e-7 # thrust coeff (Nm/(rad/s)/(rad/s)) (1.79 e -9 Nm/RPM/RPM)
    
            
            tau = (1 + (0.5 - tyMean2[ 9]) * maxChange / 100) * 0.015 # Value for second order system for Motor dynamics

            

            kp = (1 + (0.5 - tyMean2[ 10]) * maxChange / 100) * 1 # Value for second order system for Motor dynamics
            damp = (1 + (0.5 - tyMean2[ 11]) * maxChange / 100) * 1 # Value for second order system for Motor dynamics

            

            #### Early stopping
            earlyStop = False
            if earlyStop :
                if best_loss > newLoss :
                    best_loss = newLoss
                    count = 0
                else :
                    count = count + 1
                    if count > 10:
                        break
                if np.abs(best_loss - newLoss) < 0.001*best_loss and best_loss < 2:
                    count2 = count2 + 1
                elif np.abs(best_loss - newLoss) < 0.0005*best_loss and best_loss >= 2 :
                    count3 = count3 + 1
                else:
                    count2 = 0
                if count2 > 5 :
                    break

                if count3 > 10 :
                    break


            ####

                

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
                
               
                
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_lossV2,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test MARD: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_lossV2,test_acc
        ))
        maxChange = 95
        dxm = (1+(0.5-tyMean2[0]) * maxChange / 100) * 0.16 # arm length (m)
        dym = (1 + (0.5 - tyMean2[1]) * maxChange / 100) * 0.16 # arm length (m)

            

        mB = 1.2 # kg
        g = 9.81 # m/s/s
        dzm = (1 + (0.5 - tyMean2[2]) * maxChange / 100) * 0.05 # motor height (m)
        IBxx = (1 + (0.5 - tyMean2[3]) * maxChange / 100) * 0.0123 # inertia
            
           
        IByy = (1 + (0.5 - tyMean2[4]) * maxChange / 100) * 0.0123 # inertia
            
        IBzz = (1 + (0.5 - tyMean2[ 5]) * maxChange / 100) * 0.0123 # inertia
        Cd = (1 + (0.5 - tyMean2[ 6]) * maxChange / 100) * 0.1

            
        kTh = (1 + (0.5 - tyMean2[ 7]) * maxChange / 100) * 1.076e-5 # thrust coeff (N/(rad/s)/(rad/s)) (1.18 e -7 N/RPM/RPM)
        kTo = (1 + (0.5 - tyMean2[ 8]) * maxChange / 100) * 1.632e-7 # thrust coeff (Nm/(rad/s)/(rad/s)) (1.79 e -9 Nm/RPM/RPM)
    
            
        tau = (1 + (0.5 - tyMean2[ 9]) * maxChange / 100) * 0.015 # Value for second order system for Motor dynamics

            

        kp = (1 + (0.5 - tyMean2[ 10]) * maxChange / 100) * 1 # Value for second order system for Motor dynamics
        damp = (1 + (0.5 - tyMean2[ 11]) * maxChange / 100) * 1 # Value for second order system for Motor dynamics
        

        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
                best_epoch,
                train_loss, train_acc,
                valid_loss, valid_acc,
                test_loss, test_acc
        ))

        file_path = 'outputV2GPUV3.csv'

        # Define the column labels and values
        columns = ['numID', 'test_loss', 'test_mard', 'mB', 'g','dxm', 'dym', 'dzm','IBxx', 'IByy', 'IBzz', 'Cd', 'kTh', 'kTo', 'tau', 'kp', 'damp']
        values = [numID, test_loss, test_lossV2, mB, g, dxm, dym, dzm, IBxx, IByy, IBzz, Cd, kTh, kTo, tau, kp, damp]  # Replace with your float values
        if test_loss < 40:
            # Check if the file exists
            file_exists = os.path.exists(file_path)

            # Open the file in append mode if it exists, or write mode if it doesn't
            with open(file_path, mode='a' if file_exists else 'w', newline='') as file:
                writer = csv.writer(file)

                # If the file doesn't exist, write the header
                if not file_exists:
                    writer.writerow(columns)

                # Write the new row of values
                writer.writerow(values)

            print(f"Values added to {'existing' if file_exists else 'new'} CSV file.")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="ltc")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=32,type=int)
    parser.add_argument('--epochs',default=80,type=int)
    parser.add_argument('--id',default=1,type=int)
    args = parser.parse_args()

    numID = args.id
    
    test_x_x = np.loadtxt(f"data/har/UCI HAR Dataset/trainSim/xData.txt")
    Nloop = test_x_x.shape[1] # convert to 1 if batchsize > 1
    print("Nloop main {}".format(Nloop))

    har_data = HarData()
    model = HarModel(model_type = args.model,model_size=args.size)

    model.fit(har_data,epochs=args.epochs,log_period=args.log)

    print(model.y)

    #breakpoint()

