import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler #0 ortalamalı, 1stdli şekilde
from sklearn.preprocessing import LabelBinarizer #one-hot encoding yapar

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Layer, Flatten
from tensorflow.keras import backend as K

import warnings
warnings.filterwarnings("ignore")


iris = load_iris()
x= iris.data
y= iris.target

#one-hot encoding
label_binarizer = LabelBinarizer()
y_encoded=label_binarizer.fit_transform(y)

#standardizasyon
scaler = StandardScaler()
x_scaled= scaler.fit_transform(x)

#train_test_split
x_train,x_test,y_train,y_test = train_test_split(x_scaled,y_encoded,test_size=0.2,random_state=42)

class RBFLayer(Layer): #RBFLayer Keras'ın Layer sınıfından miras alır
    
    def __init__(self,units,gamma,**kwargs):
        """
            constructor,
            katmanin genel ozelliklerini baslatmak için gereklidir
        """  
        super(RBFLayer, self).__init__(**kwargs) #layer sınıfının init metodunu çağırır
        #katmanın genel özelliklerini baslatmak için gereklidir
        self.units=units #rbf katmanında gizli nöron sayısı
        self.gamma=K.cast_to_floatx(gamma) #rbf fonksiyonu yayılım  parametresi, yayılım duyarlılığı diyebiliriz
        #gamma'yı keras float32 türüne çevirmiş olduk
    
    
    def build(self,input_shape):
        """
            build metodu katmanin ağırlıklarını tanımlar
            bu metot, keras tarafından katman ilk defa bir input aldığında otomatik olarak çağrılır
        """ 
        
        #add_weight = kerasta egitilebilecek ağırlıkları tanımlamak için kullanılır
        self.mu=self.add_weight(name="mu",
                                shape=(int(input_shape[1]),self.units), #shape= ağırlıkların boyutu, input_shape[1]= giriş verisinin boyutu, self.units = merkezlerin sayısı
                                initializer="uniform", #ağırlıkların başlangıç değeri belirlenir
                                trainable=True, #ağırlıklar eğitilebilir
                                )
        super(RBFLayer,self).build(input_shape) #layer sınıfının build metodu çağrılır
        
    
    def call(self, inputs):
        """
            katman cağırıldığında yani forward propagation sırasında çalışır,
            girdiyi alır, çıktıyı hesaplar
        """
        
        #inputlar ve merkezler arası fark hesaplanır
        diff = K.expand_dims(inputs) - self.mu #K.expand_dims girdiye bir boyutr ekler      
        l2=K.sum(K.pow(diff,2),axis=1)
        res = K.exp(-1*self.gamma*l2)
        
        return res
        
    def compute_output_shape(self,input_shape):
        """
            bu metod katmanın çıktısının şekli hakkında bilgi verir
            keras'ın hyardımcı fonksiyonlarından birisidir
        """
        return (input_shape[0],self.units)  #çıktının şekli (num samples,num units)
    

def build_model():
    model= Sequential()
    
    model.add(Flatten(input_shape=(4,))) #giriş verisi düzleştir
    model.add(RBFLayer(10,0.5)) #RBF Katmanı 10 nöron ve 0.5 gamma ile eklendi
    model.add(Dense(3,activation="softmax"))
    
    #compile
    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    
    return model


#model oluşturma
model= build_model()

history= model.fit(x_train,y_train,
                   epochs=250,
                   batch_size=4,
                   validation_split=0.3,
                   verbose=1)

loss,accuracy = model.evaluate(x_test,y_test)
print(f"Test loss:{loss}, Test accuracy:{accuracy}")

#görselleştirme
plt.figure()
plt.subplot(1,2,1)
plt.plot(history.history["loss"],label="Training Loss")
plt.plot(history.history["val_loss"],label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.figure()
plt.subplot(1,2,2)
plt.plot(history.history["accuracy"],label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
