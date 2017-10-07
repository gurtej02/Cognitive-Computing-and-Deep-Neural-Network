
from keras.optimizers import SGD, Adam, RMSprop

"""In all the configrations the values follow the order BATCH_SIZE, NB_EPOCH, NB_CLASSES, VALIDATION_SPLIT, LEARNING_RATE, OPTIM, LOSS_FUNC, ACT_FUNC, DROPOUT, MODEL_NAME,
MODEL_WEIGHT, NEURONS"""


"""
configration_1 is the base code configration
"""
configration_1 = [32,20,10, 0.2,.0001, RMSprop(),"categorical_crossentropy", "relu",0.25,'configration_1_cifar10_architecture.json'
                 ,'configration_1_cifar10_weights.h5', 512]




"""
configration_2 is the same configration as configration_1, the only difference is we will be applying it to the shallow model
"""
configration_2 = [32,20,10, 0.2,.0001, RMSprop(),"categorical_crossentropy", "relu",0.25,'configration_2_cifar10_architecture.json'
                 ,'configration_2_cifar10_weights.h5', 512]


"""

Modifications over previous configrations:

Batch size is incresed to 128
SGD is used as the optimizer
number of neurons decreased to 256 for hidden layers
Using deeper neural network
Learning rate increased to 0.001

"""
configration_3 = [128,20,10, 0.2,.001, SGD(),"categorical_crossentropy", "relu",0.25,'configration_3_cifar10_architecture.json'
                 ,'configration_3_cifar10_weights.h5', 256]


"""
Using Adam optimizer
Activation function is still relu 
We will increase the number of epochs to 30
Batch size is kept as 128
loss function is kept as "categorical_crossentropy" for being able to compare accuracies
CNN network with multiple hidden layer is used 
We will increase dropout to 0.35 and see there is any impact
"""
configration_4 = [128,30,10, 0.2,.001, Adam(),"categorical_crossentropy", "relu",0.35,'configration_4_cifar10_architecture.json'
                 ,'configration_4_cifar10_weights.h5', 256]



"""
Configration for 12000 image tranning
Parameters are kept same as configrations_4 as it was the model with best accuracy

"""

configration_5 = [128,30,10, 0.2,.001, Adam(),"categorical_crossentropy", "relu",0.35,'configration_5_cifar10_architecture.json'
                 ,'configration_5_cifar10_weights.h5', 256]
