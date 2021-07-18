#%%
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.2/bin")
import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, Flatten, Dense, Lambda, Reshape
from keras.models import Model 
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from tensorflow.keras import utils as np_utils
from tensorflow.keras.layers import Concatenate as concat
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, train_test_split
from sklearn.manifold import TSNE

#%%

#Pick Directory
os.chdir("C:/Users/Joshua")
#Load MNIST
x = np.load("X3.npy")
y = np.load("y3.npy")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
del(x)
del(y)

#Normalize
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = x_train / 255
x_test = x_test / 255

#Reshape
img_width = x_train.shape[1]
img_height = x_train.shape[2]
num_channels = 3  
x_train = x_train.reshape(x_train.shape[0], img_height, img_width, num_channels)
x_test = x_test.reshape(x_test.shape[0], img_height, img_width, num_channels)
input_shape_x = (img_height, img_width, 1)
x_train_mono = np.mean(x_train, axis = -1)
x_train_mono = x_train_mono.reshape(x_train.shape[0], img_height, img_width, 1)  
x_test_mono = np.mean(x_test, axis = -1)
x_test_mono = x_test_mono.reshape(x_test.shape[0], img_height, img_width, 1) 

#convert labels to one-hot vectors
y_train =np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

input_shape_y = y_train.shape[1]

latent_dim = 20 #Number of latent dim parameters

input_img = Input(shape=input_shape_x, name="encoder_input")
input_cond = Input(shape=input_shape_y, name="conditional_input")

#%%
x = Conv2D(256, 3, padding= "same", activation = "relu")(input_img)
x = Conv2D(128, 3, padding= "same", activation = "relu", strides=(2))(x)
x = Conv2D(64, 3, padding= "same", activation = "relu", strides=(2))(x) #Change strides to 3 for X2
x = Conv2D(32, 3, padding= "same", activation = "relu", strides=(2))(x) #Change strides to 3 for X2
conv_shape = K.int_shape(x) #Shape of conv to be provided to decoder
x = Flatten()(x)
x = Dense(32, activation = "relu")(x)

#%%
#Two outputs for latent mean + STDev
z_mu = Dense(latent_dim, name= "latent_mu")(x)
z_sigma = Dense(latent_dim, name = "latent_sigma")(x)

#Define sampling function to sample from the distribution
def sample_z(args):
    z_mu, z_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mu)[0], latent_dim), mean=0., stddev=0.1)
    return z_mu + K.exp(z_sigma / 2) * epsilon

#Sample vector from the latent distribution
z = Lambda(sample_z, output_shape = (latent_dim, ), name = "z")([z_mu, z_sigma])
#add y_labels to z np.concantenate

z_concat = concat(axis=-1)([z, input_cond]) #concatenate after Dense layer 

#%%
#Define encoder
encoder = Model([input_img, input_cond], [z_mu, z_sigma, z, input_cond], name = "encoder")
print(encoder.summary())

#Define decoder inputs
latent_input = Input(shape=(latent_dim, ), name = "decoder_input")
decoder_cond = Input(shape=input_shape_y, name="decoder_conditional_input")
decoder_input = concat(axis=1)([latent_input, decoder_cond])

#Shape needs to start with something that can remapped to the original input image
x = Dense(conv_shape[1]*conv_shape[2]*conv_shape[3], activation = "relu")(decoder_input)
#reshape x to the same shape as the final layer in the encoder
x = Reshape((conv_shape[1], conv_shape[2], conv_shape[3]))(x)
#upscale back to original shape

x = Conv2DTranspose(32, 3, padding="same", activation = "relu", strides=(2))(x) 
x = Conv2DTranspose(64, 3, padding="same", activation = "relu", strides=(2))(x) 
x = Conv2DTranspose(128, 3, padding="same", activation = "relu", strides=(2))(x)
x_output = Conv2DTranspose(256, 3, padding="same", activation = "relu")(x)
#Can add more Conv2dTranspose layers using sigmoid activation
x_output = Conv2DTranspose(1, 3, padding="same", activation = "sigmoid", name = "decoder_output")(x)

#Define and summarize decoder model
decoder = Model([latent_input, decoder_cond], x_output, name="decoder")
decoder.summary()

#%%
#apply the decoder to the latent sample
z_decoded = decoder(encoder([[input_img, input_cond]])[2:])

#Call model
cvae = Model([input_img, input_cond], z_decoded, name = "cvae")

#Define loss function
recon_loss_elementwise = keras.losses.binary_crossentropy(input_img, z_decoded)
recon_loss = tf.reduce_mean(recon_loss_elementwise)

BETA = 0.0001

kl_loss_elementwise = 1 + z_sigma - tf.pow(z_mu,2) - tf.exp(z_sigma)
kl_loss = -0.5 * tf.reduce_mean(kl_loss_elementwise) * BETA

cvae.add_loss(kl_loss)
cvae.add_metric(kl_loss, name='kl_loss')

cvae.add_loss(recon_loss)
cvae.add_metric(recon_loss, name='recon_loss')

#Compile VAE
opt = keras.optimizers.Adam(learning_rate=0.001)
cvae.compile(optimizer = opt, loss= None)
cvae.summary()

cvae.fit([x_train_mono, y_train], None, epochs =  20, batch_size =16, validation_split =0.2)

#%%
def batch_predict(inputs, model, batch_size=16, verbose=True):
    """generate network predictions on large input
    by iterating through minibatches, returning one array"""
    input_size = len(inputs[0])
    num_batches = int(np.ceil(input_size / batch_size))
    mu_outputs = []
    sigma_outputs = []
    for i in range(num_batches):
        if verbose:
            print(f'Predicting batch {i+1}/{num_batches+1}')
        img_batch = inputs[0][i*batch_size:(i+1)*batch_size]
        cond_batch = inputs[1][i*batch_size:(i+1)*batch_size]
        mu,sigma,_,_ =model([img_batch, cond_batch])
        mu_outputs.append(mu)
        sigma_outputs.append(sigma)
    return np.concatenate(mu_outputs, axis=0),np.concatenate(sigma_outputs, axis=0)

#%%
#how much variance across each latent dimension?
all_means, all_vars = batch_predict([x_train_mono, y_train], encoder)
var_of_means = np.var(all_means, axis=0)
var_of_vars = np.var(all_vars, axis=0)

num_used_latents = np.sum(var_of_means > 1e-1)
print(f'Number of latent dimensions with nonzero variance: {num_used_latents}')

which_latents = np.where(var_of_means > 1e-1)[0]
print(f'They are: {which_latents}')

high_var_latents = which_latents[:2]
try:
    a,b = high_var_latents
except:
    print("no high variant dimensions")
    a = 0
    b = 1

#%%
#visualisation of inputs mapped to the latent space. 
y_test_int = np.argmax(y_test, axis=1)
te_latent= encoder.predict([x_test_mono, y_test], batch_size=16)
pred_mu, pred_var, pred_z, pred_cond = te_latent[0], te_latent[1], te_latent[2], te_latent[3]
plt.figure(figsize=(10, 10))
plt.scatter(pred_z[:, a], pred_z[:, b], c=y_test_int, cmap='brg')
plt.xlabel('dim 1')
plt.ylabel('dim 2')
plt.colorbar
plt.show()

#%%
n = 10

#Create a Grid of latent variables, to be provided as inputs to decoder.predict
grid_y = np.linspace(-3, 3, n)

figure = np.zeros((img_height * n, img_width * 2, 1))

# decoder for each square in the grid
for i, yi in enumerate(grid_y):
    for j in range(2):
        z_sample = np.zeros((32,latent_dim))
        z_sample[0,a] = yi
        cond_sample = np.zeros((32,2))
        cond_sample[0,j] = 1
        x_decoded = decoder([z_sample, cond_sample])
        # digit = x_decoded[0].numpy().reshape(img_width, img_height, num_channels)
        figure[i * img_width: (i + 1) * img_width,
               j * img_height: (j + 1) * img_height] = x_decoded[0].numpy()

#%%
plt.figure()
#Reshape for visualization
fig_shape = np.shape(figure)
# figure = figure((fig_shape[0], fig_shape[1]))
plt.imshow(figure, cmap='Greys')
plt.show()

#%%
image_batch = x_train_mono[:32]
cond_batch = y_train[:32]
reconstruction = cvae([image_batch, cond_batch])
fig = plt.figure()
fig.add_subplot(1,2,1)
plt.imshow(x_train_mono[0], cmap='Greys')
fig.add_subplot(1,2,2)
plt.imshow(reconstruction[0],cmap='Greys' )
plt.show()

# %%
x_train_2_ele = x_train_mono[:1000]
y_train_2_ele = y_train[:1000]
y_train_2_ele_int = np.argmax(y_train_2_ele, axis = 1)
plot_mu, plot_sigma = batch_predict([x_train_2_ele, y_train_2_ele],encoder)
plot_z = Lambda(sample_z, output_shape = (latent_dim, ), name = "z")([plot_mu, plot_sigma])
plot_z = plot_z.numpy()
tsne = TSNE(n_components= 2, perplexity = 100, verbose = 1)
tsne_z = tsne.fit_transform(plot_z)
colours = ["red", "blue"]
for img_class in [0,1]:
    class_x = tsne_z[:,0][np.where(y_train_2_ele_int == img_class)]
    class_y = tsne_z[:,1][np.where(y_train_2_ele_int == img_class)]
    scatter = plt.scatter(class_x, class_y, c = colours[img_class], s = 3, label = ["Benign", "Malignant"][img_class])
plt.xlabel("T-SNE Dimension 1")
plt.ylabel("T-SNE Dimension 2")
plt.legend()
plt.show()


# %%