import tensorflow as tf

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from kerastuner import HyperModel, RandomSearch


(train_images,train_labels),(test_images,test_labels) = fashion_mnist.load_data()

train_images = train_images.reshape(-1,28,28,1).astype("float32") /255.0
test_images = test_images.reshape(-1,28,28,1).astype("float32") / 255.0

train_labels = to_categorical(train_labels,10)
test_labels = to_categorical(test_labels,10)

def residual_block(x , filters , kernel_size = 3 , stride = 1):
    
    shortcut = x
    
    # 1.Conv Katmanı 
    x = Conv2D(filters,kernel_size=kernel_size,strides=stride,padding="same")(x)
    x= BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # 2.Conv Katmanı 
    x = Conv2D(filters,kernel_size=kernel_size,strides=stride,padding="same")(x)
    x= BatchNormalization()(x)
    
    #eğer girişten gelen verinin boyutu filtre sayısına eşit değilse
    if shortcut.shape[-1] != filters:
        # shortcut katmanını genişletmek için Conv Katmanı
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    #residual bağlantısı : giriş verisi ile çıkış verisini toplayalım
    x = Add()([x, shortcut])
    x= Activation('relu')(x)
    
    return x

class ResNetModel(HyperModel):
    
    def build(self,hp): # hp = hyperparameter Tuning için kullanılacak parametre
        
        inputs = Input(shape = (28,28,1))
        
        # 1.Conv Layer
        x = Conv2D(filters = hp.Int("initial_filters",min_value=32,max_value=128,step=32),
                   kernel_size=3,padding="same", activation = "relu")(inputs)
        x= BatchNormalization()(x)
        
        #residual blok ekleyelım
        for i in range(hp.Int("num_blocks",min_value=1,max_value=3,step=1)):
            x = residual_block(x,hp.Int("residual_filters_" + str(i),min_value=32,max_value=128,step=32))
        
        #sınıflandırma katmanı
        x = Flatten()(x)
        x= Dense(128,activation="relu")(x)
        outputs = Dense(10,activation="softmax")(x)
        
        model = Model(inputs,outputs)
        
        #compile model
        model.compile(optimizer = Adam(hp.Float("learning_rate",min_value=1e-4,max_value=1e-2,sampling= "LOG")),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
            
        return model
        

tuner = RandomSearch(
    ResNetModel(),
    objective= "val_accuracy",
    max_trials=2,
    executions_per_trial=1,
    directory = "resnet_hyperparameter_tuning_directory",
    project_name = "resnet_model_tuning"
)

tuner.search(train_images, train_labels, epochs = 1,validation_data = (test_images,test_labels))

#en iyi modeli alalım
best_model = tuner.get_best_models(num_models=1)[0]

#test seti ile en iyi modeli test edelim
test_loss, test_acc = best_model.evaluate(test_images,test_labels)
print(f"Test Loss: {test_loss:.3f}, Test Accuracy: {test_acc:.3f}")


