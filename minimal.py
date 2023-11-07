import numpy as np
import matplotlib.pyplot as plt
import random
import os

import keras
from keras import models
from keras import layers


print("modulo cargado...")

label={0:"paramagnetico", 1:"ferro", 2:"neel", 3:"stripe"}


def gen_spinconf(L,conf=0):
  '''
  L = size of the lattice
  conf = indicates the typo of configuration that we want to generate
  0=random
  1=ferro up
  2=ferro down
  3 = neel 1.. a
  n so on

  '''

  if conf ==1:
    #Ferro    
    red=np.ones(shape=(L,L))
  elif conf == 2:
    #Ferro
    red=-1.0*np.ones(shape=(L,L))
  elif conf == 3:
    #neel
    red=np.ones(shape=(L,L))
    for i in range(L):
      for j in range(L):
        red[i,j]*=(-1)**(i+j)
  elif conf == 4:
    #neel
    red=np.ones(shape=(L,L))
    for i in range(L):
      for j in range(L):
        red[i,j]*=(-1)**(i+j+1)
  elif conf==5:
    #stripe
    red=np.array([(-1)**j for j in range(L*L)])
    red = red.reshape(L,L)
  elif conf==6:
    #stripe
    red=np.array([(-1)**(j+1) for j in range(L*L)])
    red = red.reshape(L,L)
  elif conf==7:
    #stripe
    red=np.array([(-1)**j for j in range(L*L)])
    red = red.reshape(L,L)
    red=red.transpose()
  elif conf==8:
    #stripe
    red=np.array([(-1)**(j+1) for j in range(L*L)])
    red = red.reshape(L,L)
    red=red.transpose()
  elif conf==0:
    #paramagnetico
    red=np.array([(random.randint(0,1))*2-1 for j in range(L*L)])
    red = red.reshape(L,L)
 

  return(red.astype(int))




def generador_de_datos(c_e,L):
  """ La funcion generador_de_datos(c_e,L) nos devuelve un 
  vector train_images, el cual contiene una cantidad c_e de redes de espines de los distintos tipos de tamaño LxL. """

  
  train_images=[]
  train_labels=[]  
  

  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=1)
    train_images.append(red)
    train_labels.append(1)  

  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=2)
    train_images.append(red)
    train_labels.append(1)

  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=3)
    train_images.append(red)
    train_labels.append(2)
    
  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=4)
    train_images.append(red)
    train_labels.append(2)
    i+=1

    
  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=5)
    train_images.append(red)
    train_labels.append(3) 


  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=0)
    train_images.append(red)
    train_labels.append(0)
    
  


  # ponemos en orden aleatorio las redes dentro del vector, para que nos queden con el label correcto primero 
  # unimos las listas con zip(), las mezclamos y luego unpacking, (nos devuelve un tuple el operador zip(*) asique lo convertimos a lista)
  temp = list(zip(train_images, train_labels))
  random.shuffle(temp)
  train_images, train_labels = zip(*temp)
  train_images, train_labels= list(train_images), list(train_labels)
    
  return train_images, train_labels



# La siguiente función no es necesaria

def generador_de_datos_con_T(c_e,L,d):
  """ me genera un set de datos con una cantidad c_e de redes de espines de tamaño L*L ferromagneticas/neel/stripes. 
      d indica un desorden provocado por una "temperatura": d es el porcentaje de la cantidad de espines que van a tener un orden aleatorio en la red """
  n=int(d*L*L/100) #numero de espines aleatorios
  
  train_images=[]
  train_labels=[]  
  i=0

  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=1)
    for j in range(n):
      x=random.randint(0,L-1)
      y=random.randint(0,L-1)
      red[x,y]=(random.randint(0,1))*2-1
    train_images.append(red)
    train_labels.append(0)  

  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=2)
    for j in range(n):
      x=random.randint(0,L-1)
      y=random.randint(0,L-1)
      red[x,y]=(random.randint(0,1))*2-1
    train_images.append(red)
    train_labels.append(0)

  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=3)
    for j in range(n):
      x=random.randint(0,L-1)
      y=random.randint(0,L-1)
      red[x,y]=(random.randint(0,1))*2-1
    train_images.append(red)
    train_labels.append(1)
    
  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=4)
    for j in range(n):
      x=random.randint(0,L-1)
      y=random.randint(0,L-1)
      red[x,y]=(random.randint(0,1))*2-1
    train_images.append(red)
    train_labels.append(1)
    i+=1

  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=5)
    for j in range(n):
      x=random.randint(0,L-1)
      y=random.randint(0,L-1)
      red[x,y]=(random.randint(0,1))*2-1
    train_images.append(red)
    train_labels.append(2)
    i+=1

  for i in range(int(c_e/6)):
    red=gen_spinconf(L,conf=0)
    for j in range(n):
      x=random.randint(0,L-1)
      y=random.randint(0,L-1)
      red[x,y]=(random.randint(0,1))*2-1
    train_images.append(red)
    train_labels.append(3)
    i+=1
  
  # ponemos en orden aleatorio las redes dentro del vector, para que nos queden con el label correcto primero 
  # unimos las listas con zip(), las mezclamos y luego unpacking, (nos devuelve un tuple el operador zip(*) asique lo convertimos a lista)
  temp = list(zip(train_images, train_labels))
  random.shuffle(temp)
  train_images, train_labels = zip(*temp)
  train_images, train_labels= list(train_images), list(train_labels)
    
  return train_images, train_labels
  
  

# constructor de redes densas
def get_densa(L,neurons=512):
    model = models.Sequential()
    model.add(layers.Dense(neurons, activation='relu', input_shape=(L*L,))) #512
    model.add(layers.Dense(4, activation='softmax'))
    print(model.summary())


    model.compile(optimizer="rmsprop", 
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model




# constructor de autoencoders
def get_autoencoder(L,encoding_dim=5):

    rl2=0.01
    L_modelo=L*L

    # This is our input image
    input_img = keras.Input(shape=(L_modelo,))

    # "encoded" is the encoded representation of the input
    encoded = layers.Dense(encoding_dim, activation='relu',kernel_regularizer=keras.regularizers.l2(rl2), name="encoder")(input_img)

    # "decoded" is the lossy reconstruction of the input
    decoded = layers.Dense(L_modelo, activation='sigmoid',kernel_regularizer=keras.regularizers.l2(rl2),name="decoder")(encoded)

    # This model maps an input to its reconstruction
    autoencoder = keras.Model(input_img, decoded)

    # This model maps an input to its encoded representation
    encoder = keras.Model(input_img, encoded)

    # This is our encoded input
    encoded_input = keras.Input(shape=(encoding_dim,))

    # Retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]

    # Create the decoder model
    decoder = keras.Model(encoded_input, decoder_layer(encoded_input))



    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())


    return autoencoder, encoder, decoder



