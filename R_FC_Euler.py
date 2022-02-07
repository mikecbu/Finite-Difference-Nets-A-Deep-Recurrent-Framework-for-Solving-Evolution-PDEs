import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

gamma=1.4

sampling_number=90
time_step_size= 0.015
eps=1e-6
grids=tf.constant(np.arange(start=0.0, stop=1.0, step=1.0/sampling_number))


model=keras.Sequential(
    [
        layers.Flatten(),
        layers.Dense(400, activation="relu", kernel_initializer=keras.initializers.Identity()),
        layers.Dense(3*sampling_number, kernel_initializer=keras.initializers.Identity())
    ]
)

load=False
if load:
    function_val=model(tf.zeros([1, sampling_number]), training=False)        
    model.load_weights(".\\CNN_FP.h5")
    function_val=model(tf.zeros([1, sampling_number]), training=False)        
    for i in range(5):
        function_val=model(function_val, training=False)
        print(function_val)
    
    plt.scatter(grids, function_val)
    plt.show()
    exit(0)
    

# Output 3-tuple: rho (density), v (velocity), p (pressure)
def initial_cond(x):
    if x<=0.5:
        return 8.0, 0.0, 8.0
    else:
        return 1.0, 0.0, 1.0

def intra_tensor_integral_1D(tensor, interval_length):
    shifted_tensor=tf.roll(tensor, shift=-1, axis=0)
    integral=(shifted_tensor+tensor)[:-1]*interval_length
    return integral    

def intra_tensor_grad_1D(tensor, interval_length):
    shifted_tensor=tf.roll(tensor, shift=-1, axis=0)
    grad=tf.concat([(shifted_tensor-tensor)[:-1]/interval_length, (shifted_tensor-tensor)[-2:-1]/interval_length], axis=0)
    return grad
    
def intra_tensor_grad_2D(tensor):
    shifted_v_tensor=tf.roll(tensor, shift=1, axis=1)
    shifted_x_tensor=tf.roll(tensor, shift=1, axis=2)
    mask_x=tf.concat([tf.zeros([1]), tf.ones([x_sampling_number-2]), tf.zeros([1])], axis=0)
    mask_v=tf.concat([tf.zeros([1]), tf.ones([v_sampling_number-2]), tf.zeros([1])], axis=0)
    grad_x=tf.math.multiply((tensor-shifted_x_tensor), tf.broadcast_to(mask_x, [x_sampling_number, v_sampling_number]))
    grad_v=tf.math.multiply((tensor-shifted_v_tensor), tf.broadcast_to(tf.transpose(mask_v), [x_sampling_number, v_sampling_number]))
    return grad_x, grad_v

        
total_time_steps =20
for current_time_step in range(total_time_steps):
    print("\nStart of time step %d" % (current_time_step,))
    
    optimizer=keras.optimizers.Adam(learning_rate=0.001) 
    
    if current_time_step==0:
        initial_values=np.zeros((3, sampling_number))
        for n_x in range(sampling_number):
            initial_values[0, n_x], initial_values[1, n_x], initial_values[2, n_x]=initial_cond(n_x/sampling_number)
            
        plt.figure()
        plt.subplot(3,1,1)
        plt.title("t={:0.4f}".format(current_time_step*time_step_size))
        plt.plot(grids,initial_values[0,:],'b.')
        plt.ylabel('Density')
        plt.xlim([0,1.0])
        plt.subplot(3,1,2)
        plt.plot(grids,initial_values[1,:],'b.')
        plt.ylabel('Velocity')
        plt.xlim([0,1.0])
        plt.ylim([-0.3, 0.3])
        plt.subplot(3,1,3)
        plt.plot(grids,initial_values[2,:],'b.')
        plt.xlim([0,1.0])
        plt.ylabel('Pressure')
       
        grap1=str("result/grap111.png")
        plt.savefig(grap1)
        

        initial_values=tf.reshape(initial_values, [1, 3, sampling_number])
        initial_values=tf.dtypes.cast(initial_values, tf.float32)
        previous_function_val=tf.reshape(model(initial_values, training=True), [1, 3, sampling_number])
    else:
        previous_function_val=function_val
    
    # Iterate over the batches of the dataset.
    for step in range(1000):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.            
        
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            
            function_val = tf.reshape(model(previous_function_val, training=True), [1, 3, sampling_number])
            
            
                        
            u1_t1=previous_function_val[0][0]
            u1_t2=function_val[0][0]
            fu1_t1=tf.math.multiply(previous_function_val[0][0], previous_function_val[0][1])
            fu1_t2=tf.math.multiply(function_val[0][0], function_val[0][1])
            
            u2_t1=fu1_t1
            u2_t2=fu1_t2
            fu2_t1=tf.math.multiply(fu1_t1, previous_function_val[0][1])+previous_function_val[0][2]
            fu2_t2=tf.math.multiply(fu1_t2, function_val[0][1])+function_val[0][2]
            
            E_t1=previous_function_val[0][2]/(gamma-1)+0.5*tf.math.multiply(fu1_t1, previous_function_val[0][1])
            E_t2=function_val[0][2]/(gamma-1)+0.5*tf.math.multiply(fu1_t2, function_val[0][1])
            
            u3_t1=E_t1
            u3_t2=E_t2
            fu3_t1=tf.math.multiply(previous_function_val[0][1], E_t1+previous_function_val[0][2])
            fu3_t2=tf.math.multiply(function_val[0][1], E_t2+function_val[0][2])

            derivation_tensor_1=intra_tensor_integral_1D(u1_t2-u1_t1, 1.0/sampling_number)-(fu1_t2+fu1_t1)[:-1]*time_step_size+(fu1_t2+fu1_t1)[1:]*time_step_size                 
            derivation_tensor_2=intra_tensor_integral_1D(u2_t2-u2_t1, 1.0/sampling_number)-(fu2_t2+fu2_t1)[:-1]*time_step_size+(fu2_t2+fu2_t1)[1:]*time_step_size                 
            derivation_tensor_3=intra_tensor_integral_1D(u3_t2-u3_t1, 1.0/sampling_number)-(fu3_t2+fu3_t1)[:-1]*time_step_size+(fu3_t2+fu3_t1)[1:]*time_step_size                 
            
                                        
            # Spectual Reflection Boundary
            # loss_value = tf.reduce_sum(tf.math.multiply(derivation_tensor, derivation_tensor))+\
                            # tf.reduce_sum(tf.math.multiply(tf.roll(tf.transpose(function_val[0])[0], axis=0, shift=int(v_sampling_number/2))-tf.transpose(function_val[0])[0], tf.roll(tf.transpose(function_val[0])[0], axis=0, shift=int(v_sampling_number/2))-tf.transpose(function_val[0])[0]))+\
                            # tf.reduce_sum(tf.math.multiply(tf.roll(tf.transpose(function_val[0])[-1], axis=0, shift=int(v_sampling_number/2))-tf.transpose(function_val[0])[-1], tf.roll(tf.transpose(function_val[0])[-1], axis=0, shift=int(v_sampling_number/2))-tf.transpose(function_val[0])[-1]))-\
                            # tf.reduce_sum(tf.clip_by_value(function_val, clip_value_min=-100, clip_value_max=0))+\
                            # abs(tf.reduce_sum(function_val)/(x_sampling_number*v_sampling_number)-7.2)


            # Periodic Boundary
            loss_value =  tf.reduce_sum(tf.math.multiply(derivation_tensor_1, derivation_tensor_1))+\
                                tf.reduce_sum(tf.math.multiply(derivation_tensor_2, derivation_tensor_2))+\
                                tf.reduce_sum(tf.math.multiply(derivation_tensor_3, derivation_tensor_3))+\
                                900*abs(function_val[0][0][0]-8.0)+3600*abs(function_val[0][0][-1]-1.0) +\
                                2200*(abs(function_val[0][1][0])+abs(function_val[0][1][-1]))+\
                                2200*(abs(function_val[0][2][0]-8.0)+abs(function_val[0][2][-1]-1.0))
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.title("t={:0.4f}".format((current_time_step+1)*time_step_size))
    plt.plot(grids,function_val[0][0],'b.')
    plt.ylabel('Density')
    plt.xlim([0,1.0])
    plt.subplot(3,1,2)
    plt.plot(grids,function_val[0][1],'b.')
    plt.ylabel('Velocity')
    plt.xlim([0,1.0])
    plt.ylim([-1.5, 1.5])
    plt.subplot(3,1,3)
    plt.plot(grids,function_val[0][2],'b.')
    plt.xlim([0,1.0])
    plt.ylabel('Pressure')
    grap2=str("result/grap222-"+str(current_time_step)+".png")
    plt.savefig(grap2)
    plt.close("all") 
    
    print(
        "Training loss (for one batch) at time %f: %.4f"
        % (current_time_step*time_step_size, float(loss_value))
        )

model.save("CNN_FP.h5")