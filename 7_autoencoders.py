#libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

from skimage.metrics import structural_similarity as ssim

(x_train,_),(x_test,_) = fashion_mnist.load_data()

# veri normalizasyonu
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#veriyi düzleştir 28x28 boyutunu 784 boyutunda vektöre dönüştür
x_train= x_train.reshape((len(x_train),np.prod(x_train.shape[1:])))
x_test= x_test.reshape((len(x_test),np.prod(x_test.shape[1:])))

#autoencoders için model mimarilerinin tanımlanması
input_dim= x_train.shape[1] # giriş boyutu (784)
encoding_dim = 64 #latent boyutu ( daha kucuk boyutta temsil)


#encoder kısmının inşa edilmesi
input_image = Input(shape=(input_dim,)) #girdi boyutunu belirliyoruz
encoded = Dense(256, activation="relu")(input_image) #ilk gizli katman (256 nöron)
encoded = Dense(128, activation="relu")(encoded) #ikinci gizli katman (128 nöron)
encoded = Dense(encoding_dim, activation="relu")(encoded) #sıkıştırma katmanı


#decoder kısmının inşa edilmesi
decoded = Dense(128, activation="relu")(encoded) #ilk genişletme katmanı
decoded = Dense(256, activation="relu")(decoded) #ikinci genişletme katmanı
decoded = Dense(input_dim, activation="sigmoid")(decoded) #(çıktı) son genişletme katmanı (784 boyutlu)


#autoencoders oluşturma = encoder + decoder
autoencoder = Model(input_image,decoded) # girişten çıkışa tüm yapı tanımlandı

#modelin compile edilmesi
autoencoder.compile(optimizer=Adam(),
                    loss="binary_crossentropy")

#modelin train edilmesi
history = autoencoder.fit(x_train,x_train,
                          epochs=50,
                          batch_size=64,
                          shuffle=True, #eğitim verilerini karıştır
                          validation_data=(x_test,x_test),
                          verbose=1)

#Model Testi

#modeli encoder ve decoder olarak ikiye ayır
encoder = Model(input_image, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-3](encoded_input)
decoder_layer2 = autoencoder.layers[-2](decoder_layer1)
decoder_output = autoencoder.layers[-1](decoder_layer2)

decoder = Model(encoded_input, decoder_output)

encoded_images = encoder.predict(x_test) #latent temsili elde ederiz
decoded_images = decoder.predict(encoded_images) #latent temsili orjinal forma geri döndürür

#orijinal ve yeniden yapılandırılmış(decoded_images) görüntüleri görselleştir
n=10
plt.figure(figsize=(20, 4))
for i in range(n):
    
    #orjinal görüntü
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28,28), cmap = "gray") #orijinal görüntü boyutuna çevrilir
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    #decoded edilmiş yeniden yapılandırılmış görüntü için görselleştirme
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28,28),cmap="gray")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    

#SSIM Skorlarını Hesaplama
def compute_ssim(orijinal, reconstructured):
    """
        her iki görüntü arasında ssim skoru (0-1) arasında çıkar
    """
    orijinal = orijinal.reshape(28,28)
    reconstructured=reconstructured.reshape(28,28)
    return ssim(orijinal, reconstructured, data_range=1)

ssim_score = []

#ilk 100 tanesini hesaplayalım
for i in range(100):
    
    orijinal_img = x_test[i]
    reconstructured_img =decoded_images[i]
    score = compute_ssim(orijinal_img,reconstructured_img)
    ssim_score.append(score)
    
average_ssim = np.mean(ssim_score)
print("SSIM: ",average_ssim)