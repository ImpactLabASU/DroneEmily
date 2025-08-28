import numpy as np
import pandas as pd
import os
import time
import gc
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Run on GPU 0

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU
import argparse

def cut_in_sequences(x,y,seq_len,inc=1):

    sequences_x = []
    sequences_y = []

    for s in range(0,x.shape[0] - seq_len,inc):
        start = s
        end = start+seq_len
        sequences_x.append(x[start:end])
        sequences_y.append(y[start:end])

    return np.stack(sequences_x,axis=1),np.stack(sequences_y,axis=1)

class HarData:

    def __init__(self,seq_len=16):
        tf.compat.v1.disable_eager_execution()
        tf.disable_v2_behavior()
        train_x = np.loadtxt("data/har/UCI HAR Dataset/train/X_train.txt")
        train_y = (np.loadtxt("data/har/UCI HAR Dataset/train/y_train.txt")-1).astype(np.int32)
        test_x = np.loadtxt("data/har/UCI HAR Dataset/test/X_test.txt")
        test_y = (np.loadtxt("data/har/UCI HAR Dataset/test/y_test.txt")-1).astype(np.int32)

        train_x,train_y = cut_in_sequences(train_x,train_y,seq_len)
        test_x,test_y = cut_in_sequences(test_x,test_y,seq_len,inc=8)
        print("Total number of training sequences: {}".format(train_x.shape[1]))
        permutation = np.random.RandomState(893429).permutation(train_x.shape[1])
        valid_size = int(0.1*train_x.shape[1])
        print("Validation split: {}, training split: {}".format(valid_size,train_x.shape[1]-valid_size))

        self.valid_x = train_x[:,permutation[:valid_size]]
        self.valid_y = train_y[:,permutation[:valid_size]]
        self.train_x = train_x[:,permutation[valid_size:]]
        self.train_y = train_y[:,permutation[valid_size:]]

        self.test_x = test_x
        self.test_y = test_y
        print("Total number of test sequences: {}".format(self.test_x.shape[1]))

    def iterate_train(self,batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size

        for i in range(total_batches):
            start = i*batch_size
            end = start + batch_size
            batch_x = self.train_x[:,permutation[start:end]]
            batch_y = self.train_y[:,permutation[start:end]]
            yield (batch_x,batch_y)

class HarModel:

    def __init__(self,model_type,model_size,learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,561])
        self.target_y = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None,None])

        self.model_size = model_size
        head = self.x

    
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


        self.y = tf.compat.v1.layers.Dense(6,activation=None)(head)
        print("logit shape: ")
        print(str(self.y.shape))
        print("self.y: ")
        print(self.y)
        print(self.target_y)
    
        self.loss = tf.reduce_mean(tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        
        
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        # Configure GPU memory growth to prevent memory issues
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        
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

        best_valid_accuracy = 0
        best_valid_stats = (0,0,0,0,0,0,0)
        self.save()
        
        # Start timing training
        training_start_time = time.time()
        total_testing_time = 0.0
        early_stopped = False
        
        for e in range(epochs):
            # Training phase FIRST - populate losses and accs arrays
            losses = []
            accs = []
            
            for batch_x,batch_y in gesture_data.iterate_train(batch_size=16):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x:batch_x,self.target_y: batch_y})
                if(not self.constrain_op is None):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)
            
            # Check for early stopping condition AFTER training
            avg_train_loss = np.mean(losses)
            avg_train_acc = np.mean(accs)
            
            if avg_train_loss <= 0.1:
                print(f"ðŸŽ‰ Early stopping triggered! Training loss ({avg_train_loss:.4f}) <= 0.1 at epoch {e}")
                early_stopped = True
                break  # Exit immediately when early stopping is triggered

            # Testing/validation phase AFTER training
            if(verbose and e%log_period == 0):
                # Start timing testing/validation
                testing_start_time = time.time()
                
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.test_x,self.target_y: gesture_data.test_y})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.valid_x,self.target_y: gesture_data.valid_y})
                
                # Add to total testing time
                testing_end_time = time.time()
                total_testing_time += (testing_end_time - testing_start_time)
                
                # Accuracy metric -> higher is better
                if(valid_acc > best_valid_accuracy and e > 0):
                    best_valid_accuracy = valid_acc
                    best_valid_stats = (
                        e,
                        avg_train_loss,avg_train_acc*100,
                        valid_loss,valid_acc*100,
                        test_loss,test_acc*100
                    )
                    self.save()

                elapsed_time = time.time() - training_start_time
                print("Epochs {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}% (elapsed: {:.1f}s)".format(
                    e,
                    avg_train_loss,avg_train_acc*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100,
                    elapsed_time
                ))
                
            # Break for infinite loss
            if(e > 0 and (not np.isfinite(avg_train_loss))):
                break
            
            # Clear memory to prevent GPU memory issues
            del losses, accs
            gc.collect()  # Force garbage collection
        
        # Calculate and display timing results
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time
        pure_training_time = total_training_time - total_testing_time
        
        # Format timing output
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d} ({seconds:.2f}s)"
        
        print(f"\nâ±ï¸  Timing Results:")
        print(f"    Total training time: {format_time(total_training_time)}")
        print(f"    Pure training time: {format_time(pure_training_time)}")
        print(f"    Total testing time: {format_time(total_testing_time)}")
        print(f"    Early stopping: {'Yes' if early_stopped else 'No'}")
        if early_stopped:
            print(f"    Training stopped early due to low training loss (â‰¤ 0.1)")
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))

if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument('--model',default="lstm")
    parser.add_argument('--log',default=1,type=int)
    parser.add_argument('--size',default=256,type=int)
    parser.add_argument('--epochs',default=1,type=int)
    args = parser.parse_args()


    har_data = HarData()
    model = HarModel(model_type = args.model,model_size=args.size)

    model.fit(har_data,epochs=args.epochs,log_period=args.log)



