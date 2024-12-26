from keras.api import layers, Input, Sequential, optimizers
from keras.api.losses import BinaryCrossentropy
from keras.api.models import Model
import tensorflow as tf

def load_cpmp_model(generate_model, model_file: str, S: int, H: int, ) -> Model:
    device_name = tf.test.gpu_device_name()
    with tf.device(device_name):
        model=generate_model(S=S, H=H)
        model.compile(
          loss=BinaryCrossentropy(),
          optimizer=optimizers.Adam(learning_rate=0.001),
          metrics=['mse']
        )
    try:
        model.load_weights(model_file)
    except:
        raise RuntimeError("Invalid model")
    return model

def create_cpmp_model(generate_model, S: int, H: int) -> Model:
    device_name = tf.test.gpu_device_name()
    print("device_name", device_name)
    with tf.device(device_name):
        Fmodel=generate_model(S=S, H=H)
        Fmodel.compile(
              loss=BinaryCrossentropy(),
              optimizer=optimizers.Adam(learning_rate=0.001),
              metrics=['mse']
        )
    return Fmodel

def generate_model2(S: int, H: int) -> Model:
    x = Input(shape=(S*(H+1)+2*(S*(S-1)),)) #recibe el estado + tipo de movs

    sensors = []
    for i in range(S): sensors.append(x[:,i*S:i*S+H+1])

    sensor_model2 = Sequential([
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu')
    ])

    ## state encoding
    sensors_encodings=[]
    for i in range(S): 
        sensors_encodings.append(sensor_model2(sensors[i]))
    state_encoding = layers.Average()(sensors_encodings)

    sensor_model = Sequential([
    layers.Dense(256, activation='relu'),
    #layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
    ])

    k=0
    pairwise_encodings=[]
    for i in range(S): 
        for j in range(S): 
            if i==j: continue
            pairwise_encodings.append(sensor_model(layers.Concatenate()([sensors[i],sensors[j],state_encoding,x[:,S*(H+1)+2*k:S*(H+1)+2*k+2]])))
            k+=1

    h = layers.Concatenate()(pairwise_encodings)

    model = Model(inputs=x, outputs=[h])

    return model

def generate_model(S: int, H: int) -> Model:
    model = tf.keras.Sequential()

    model.add(layers.Dense(256, activation='relu',
                            input_shape=(S*(H+1)+2*(S*(S-1)),)))

    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(S*(S-1), activation='sigmoid'))
    return model