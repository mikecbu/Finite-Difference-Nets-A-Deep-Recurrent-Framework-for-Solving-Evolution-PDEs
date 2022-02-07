import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import math
import matplotlib.pyplot as plt

sampling_number=100
time_step_size=0.05
eps=1e-6
grids=np.arange(start=0.0, stop=1.0, step=1.0/sampling_number)


model=keras.Sequential(
    [
        layers.Dense(sampling_number, activation="relu"),
        layers.Dense(sampling_number, activation="relu"),
        layers.Dense(sampling_number)
    ]
)

load=False
if load:
    function_val=model(tf.zeros([1, sampling_number]), training=False)        
    model.load_weights(".\\CNN.h5")
    function_val=model(tf.zeros([1, sampling_number]), training=False)        
    for i in range(5):
        function_val=model(function_val, training=False)
        print(function_val)
    
    plt.scatter(grids, function_val)
    plt.show()
    exit(0)
    
#Solve transport equnation u_t+au_x=0, in the set x\in [0,1] and t\in [0,1]
a=-1.0

# Boundary condition at x=1
def boundary_cond(t):
    if t<=1.0:
        return math.sin(t*math.pi)**2
    else:
        return 0.0
    
def true_solution(x, t):
    if x+t<=1.0:
        return 0.0
    else:
        return boundary_cond(t-1.0+x)

def intra_tensor_grad_1D(tensor):
    shifted_tensor=tf.roll(tensor, shift=-1, axis=0)
    mask=tf.concat([tf.ones([1]), tf.ones([tensor.shape[0]-2]), tf.zeros([1])], axis=0)
    grad=tf.math.multiply((shifted_tensor-tensor), mask)
    return grad

        
total_time_steps = 30
for current_time_step in range(total_time_steps):
    print("\nStart of time step %d" % (current_time_step,))
    
    optimizer=keras.optimizers.Adam(learning_rate=0.01)

    
    
    
    if current_time_step==0:
        previous_function_val=model(tf.zeros([1, sampling_number]), training=True)
        model.summary()
    else:
        previous_function_val=function_val
    
    # Iterate over the batches of the dataset.
    for step in range(500):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.            
        
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            
            function_val = model(previous_function_val, training=True)
            
            

            # Compute the loss value for this minibatch.
            derivation_tensor=(function_val-previous_function_val)/time_step_size+\
                            a*intra_tensor_grad_1D(function_val[0])*sampling_number
            loss_value = tf.reduce_sum(tf.math.multiply(derivation_tensor, derivation_tensor))/sampling_number+\
                            10*(function_val[0][-1]-boundary_cond(time_step_size*current_time_step))**2
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.title("t={:0.4f}".format((current_time_step)*time_step_size))
    plt.scatter(grids, function_val)
    

    plt.show()
    plt.clf()
    print(
        "Training loss (for one batch) at time %f: %.4f"
        % (current_time_step*time_step_size, float(loss_value))
        )

model.save("CNN.h5")

function_val=model(tf.zeros([1, sampling_number]), training=False)