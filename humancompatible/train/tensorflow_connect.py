import tensorflow as tf
from sklearn.preprocessing import StandardScaler 
#from torch.nn.utils import clip_grad_norm_ 
import os
import numpy as np


class CustomNetwork(tf.keras.Model):

    # For now the input data is passed as init parameters
    def __init__(self, model_specs):
        super(CustomNetwork, self).__init__()

        # Create a list of linear layers based on layer_sizes
        layer_sizes = model_specs[0]
        self.layer_list = []
        self.layer_norm_list = []
        self.RANDOM_SEED = 0
        for i in range(len(layer_sizes) - 1):
            dense_layer = tf.keras.layers.Dense(layer_sizes[i+1],
                                                 kernel_initializer=tf.keras.initializers.GlorotUniform(),  # Set weight initializer
                                                 bias_initializer=tf.keras.initializers.GlorotUniform())  # Set bias initializer
            #dense_layer.trainable = True  # Set trainable to True
            self.layer_list.append(dense_layer)

        #build the network    
        self.build((None, layer_sizes[0]))
    
    def build(self, input_shape):
        # Set input shape for the first layer
        self.layer_list[0].build(input_shape)
        # Call build method for other layers
        for i in range(1, len(self.layer_list)):
            self.layer_list[i].build((input_shape[0], self.layer_list[i-1].units))
        
        
    def call(self, inputs):
        x = inputs
        for layer in self.layer_list[:-1]:
            x = tf.keras.activations.relu(layer(x))
        x = tf.keras.activations.sigmoid(self.layer_list[-1](x))
        return x
    
    ######Only this loss function is used here######
    def compute_loss(self, Y, Y_hat):
        L_sum = 0.5 * tf.reduce_sum(tf.square(Y - Y_hat))
        
        m = tf.cast(tf.shape(Y)[0], tf.float32)
        L = (1. / m) * L_sum
        return L

    def bce_loss(outputs, targets):
        loss = tf.keras.losses.BinaryCrossentropy()(targets, outputs)
        return loss
    

    def get_obj(self, x, y, params):
         
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)   

        model_parameters = self.trainable_variables

        #samples = tf.constant(samples, dtype=tf.int32)
        # Update model parameters with the provided params
        for i, param in enumerate(params):
            model_parameters[i].assign(param)
        
        # Forward pass to compute predictions 
        obj_fwd = tf.reshape(self.call(x), [-1])
        
        # Check for NaN values in predictions
        if np.isnan(obj_fwd).any():
            #print("THE MINIBATCH SIZE>>>>>", minibatch)
            for i, param in enumerate(self.trainable_variables):
                print(param.name)
                print(param.numpy())
        
        # Compute loss
        fval = self.compute_loss(obj_fwd, tf.reshape(y, [-1]))
        
        return fval.numpy()
    
    def get_obj_grad(self, x, y, params):

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        y = tf.convert_to_tensor(y, dtype=tf.float32)   
        
        # Forward pass
        with tf.GradientTape() as tape:
            obj_fwd = tf.reshape(self.call(x),[-1])
            obj_loss = self.compute_loss(obj_fwd, tf.reshape(y, [-1]))
        
        # Compute gradients
        gradients = tape.gradient(obj_loss, self.trainable_variables)
        
        # Convert gradients to a flat list
        fgrad = tf.concat([tf.reshape(grad, [-1]) for grad in gradients if grad is not None], axis=0)
        
        return fgrad
    
    def get_constraint(self, x1_sensitive, y1_sensitive, x2_sensitive, y2_sensitive, params):

        x1_sensitive = tf.convert_to_tensor(x1_sensitive, dtype=tf.float32)
        y1_sensitive = tf.convert_to_tensor(y1_sensitive, dtype=tf.float32)   
        x2_sensitive = tf.convert_to_tensor(x2_sensitive, dtype=tf.float32)
        y2_sensitive = tf.convert_to_tensor(y2_sensitive, dtype=tf.float32)   
        
        sensitive_fwd1 = tf.reshape(self.call(x1_sensitive),[-1])
        sensitive_loss1 = self.compute_loss(sensitive_fwd1, tf.reshape(y1_sensitive, [-1]))
        
        sensitive_fwd2 = tf.reshape(self.call(x2_sensitive), [-1])
        sensitive_loss2 = self.compute_loss(sensitive_fwd2, tf.reshape(y2_sensitive, [-1]))

        cons_loss = (sensitive_loss1 - sensitive_loss2)
        
        return cons_loss.numpy()
    
    def get_constraint_grad(self, x1_sensitive, y1_sensitive, x2_sensitive, y2_sensitive, params):

        x1_sensitive = tf.convert_to_tensor(x1_sensitive, dtype=tf.float32)
        y1_sensitive = tf.convert_to_tensor(y1_sensitive, dtype=tf.float32)   
        x2_sensitive = tf.convert_to_tensor(x2_sensitive, dtype=tf.float32)
        y2_sensitive = tf.convert_to_tensor(y2_sensitive, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            sensitive_fwd1 = tf.reshape(self.call(x1_sensitive),[-1])
            sensitive_loss1 = self.compute_loss(sensitive_fwd1, tf.reshape(y1_sensitive, [-1]))

            sensitive_fwd2 = tf.reshape(self.call(x2_sensitive),[-1])
            sensitive_loss2 = self.compute_loss(sensitive_fwd2, tf.reshape(y2_sensitive, [-1]))
            
            cons_loss = (sensitive_loss1 - sensitive_loss2)
         

            cons_gradients = tape.gradient(cons_loss, self.trainable_variables)
        
        max_norm = 0.5

        # Compute the global L2-norm of the gradients
        global_norm = tf.linalg.global_norm(cons_gradients)

        # If global_norm exceeds max_norm, clip gradients
        if global_norm > max_norm:
            # Clip gradients to ensure that their global L2-norm does not exceed max_norm
            cons_gradients = [tf.clip_by_norm(grad, max_norm) for grad in cons_gradients]
        else:
            cons_gradients = cons_gradients
            
        #cons_gradients = [female_grad - male_grad for female_grad, male_grad in zip(female_gradients, male_gradients)]
        cons_gradients = tf.concat([tf.reshape(grad, (-1,)) for grad in cons_gradients], axis=0)

        return cons_gradients
    
    def to_backend(obj):
        return tf.convert_to_tensor(obj)
    
    def save_model(self, dir):
        self.save(str(dir)+'.h5')

    def get_trainable_params(self):
        nn_parameters = self.trainable_variables
        initw = [param.numpy() for param in nn_parameters]
        num_param = sum(p.size for p in initw)
        return initw, num_param
    
    def evaluate(self, x):
        return self.call(x).numpy()

def load_model(directory_path, model_file):
        model = tf.keras.models.load_model(os.path.join(directory_path, model_file))
        return model
