#deep-q learning agent class
import gym #reinforcmment learning için enviroment sağlar- gelitirme ortamı
import  numpy as np
from collections import deque #ajanın belleği içşn deque  veri yapısı
from keras.models import Sequential #nural network
from keras.layers import Dense #neural network layer
from keras.optimizers import Adam #optimizer

import random #random number generation

from tqdm import tqdm #progress bar

#dql agent class
class DQLAgent:
    
    def __init__(self, env): #parametrelei e hiperparametleri tanımla
        
        #cevrenin gözlem alanı (state) boyut sayısı
        self.state_size = env.observation_space.shape[0]
        
        #çevrede bulunan eylem sayısı
        self.action_size = env.action_space.n
        
        #gelecekteki ödüllerin indirim oranı
        self.gamma = 0.95
        
        #learning rate oranı
        self.learning_rate = 0.001
        
        #keşfetme oranı (epsilon) epsilon=1 maksimum keşif
        self.epsilon = 1.0
        
        #epsilonun her iterasyonda azalma oranı (epsilon azaldıkça daha fazla öğrenme, daha az keşif)
        self.epsilon_decay = 0.995
        
        #minimum keşfetme oranı (min epsilon)
        self.epsilon_min = 0.01 
        
        #ajanın deneyimleri  = bellek = geçmiş adımlar
        self.memory = deque(maxlen=1000)
        
        #derin öğrenme modeli inşa et
        self.model = self.build_model()

    def build_model(self): #dql sinir ağı modeli oluştur
        
        model = Sequential() #sıralı model
        model.add(Dense(48, input_dim=self.state_size, activation='relu')) #girdi katmani
        model.add(Dense(24,activation="relu")) #2.gizli katman
        model.add(Dense(self.action_size,activation="linear")) #output katmanı
        #model compile
        model.compile(loss="mse",optimizer=Adam(learning_rate=self.learning_rate))
        
        return model
        
    
    def remember(self,state,action,reward,next_state,done): #ajanın deneyimlerini bellekte sakla
        self.memory.append((state,action,reward,next_state,done))
    
    def act(self,state): #ajan eylem seçebilecek
        
        if random.uniform(0,1) <= self.epsilon:
            return env.action_space.sample() #rastgele eylem seç
        act_values= self.model.predict(state,verbose=1) #modelden eylem değerleri al
        return np.argmax(act_values[0])
        
    def replay(self,batch_size): #deneyimleri tekrar oynatarak dql ağı eğitilir
        
        #bellekte yeterince deneyim yoksa geri oynatma yapılmaz
        if len(self.memory) < batch_size:
            return
        
        #bellekten batch size büyüklüğünde random deneyim seç
        minibatch = random.sample(self.memory, batch_size)
        
        for state,action,reward,next_state,done in minibatch:
            if done: #eğer done ise oyun bitmişse
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state,verbose=0)[0])
            
            #modelin tahmin ettiği ödüller
            train_target = self.model.predict(state,verbose=0)
            
            #ajanın yaptığı eyleme göre tahmin edilen ödülü günceller
            train_target[0][action] = target
            
            #modeli eğit
            self.model.fit(state,train_target,verbose = 0) 
    
    def adaptiveEGreedy(self): #epsionun zamanla azalması, keşif-sömürü dengesi
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    

env = gym.make("CartPole-v1",render_mode="human")
agent = DQLAgent(env)

batch_size = 32 #eğitim için minibatch boyutu
episodes = 2 #epoch , simulasyonun oynatılacağı toplam sayı

for e in tqdm(range(episodes)):
    
    #ortamı sıfırla ve başlangıç durumunu al
    state = env.reset()[0] #ortamı sıfırlamak
    state = np.reshape(state,[1,4])
    
    time = 0 #zaman adımını başlat
    
    while True:
        
        #ajan eylem seçer
        action = agent.act(state)
        
        #ajanımız ortamda bu eylemi uygular ve bu eylem sonunda:
        #next_state, reward, bitiş parametresi(done) alır
        (next_state,reward,done,_,_)=env.step(action)
        next_state = np.reshape(state,[1,4])
        
        #yapmış olduğu bu eylemi ve eylem sonucunda alınan bilgileri kaydeder
        agent.remember(state,action,reward,next_state,done)
        
        #mevcut durumu günceller
        state = next_state
        
        #deneyimlerden yeniden oynatmayı başlatır replay()--> training
        agent.replay(batch_size)
        
        #epsilonu azaltır
        agent.adaptiveEGreedy()
        
        #zaman adımını arttırır
        time=time+1
        
        #eğer "done" ise donguyu kırar ve bolum biter- yeni bölüm başlar
        if done:
            print(f"\nEpisode: {e}, time: {time}")
            break

import time
trained_model = agent #eğitilen model

env = gym.make('CartPole-v1') #gym ortamı oluşturuldu
state = env.reset()[0]
state = np.reshape(state, [1, 4])

time_t = 0 #zaman adımını başlat

while True:
    env.render() #ortamı görsel olarak render et
    action = trained_model.act(state) #eğitilen modelden action gerçekleştir
    (next_state,reward,done,_,_)=env.step(action) #eylemi uygla
    next_state = np.reshape(next_state, [1, 4])
    state = next_state
    time_t += 1
    print(f"time: {time_t}")
    
    time.sleep(0.5) #yarım saniye bekle
    
    if done:
        break
print("Done")

