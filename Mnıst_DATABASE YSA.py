# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:38:22 2019

@author: Salim
"""

import keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Veri setimizi yükleyelim
fashion_mnist=keras.datasets.fashion_mnist
#Egitim ve test etmek için veri setimizi tanımladık
#train_images, test_images 28x28 boyutunda ve piksel değerleri 0 ile 255 arasında değişen NumPy dizileridir.
# train_labels,test_labels ise 0 ile 9 arasında değişen ve her biri bir giyim eşyası sınıfı ile eşleşen tam sayı dizisidir:
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
#Veri kümesi içerisindeki her bir görüntü tek bir etiket ile eşleştirilmiştir. Sınıf isimleri veri kümesi içerisinde yer almadığı için,
# daha sonra görüntüleri ekrana yazdırmak için bunları aşağıdaki gibi bir dizi içerisinde saklayalım:


class_names = ['Tişört/Üst', 'Pantolon', 'Kazak', 'Elbise', 'Mont', 
               'Sandal', 'Gömlek', 'Spor Ayakkabı', 'Çanta', 'Yarım Bot']

"""Verimiz biraz inceleyelim"""
"""print(train_images.shape)
print(len(train_labels))"""
#Veri her çalıştıgında tekrar yüklenmesin diye yorum satırı yaptım 1 kere yükledikten sonra tekrar yüklemeye gerek yok 
#Veriyi biraz görselleştirelim
#0.indexdeki egitim resmimize bakalım
"""plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()"""

#Veri setimizdeki degerleri normalize etmemiz lazım 
#Yani 0 ile 1  arasında bir deger vermemiz lazım,en yüksek RGB renk kodu 255 olduğu için hepsine 255 bölelim
traim_images=train_images/255.0
test_images=test_images/255.0
#Veri dogru formattamı inceleyelim
"""plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()"""
#Veriler bende gayet güzel çıktı
#%%
#Artık modelimizi oluşturabiliriz

model=keras.Sequential([
        keras.layers.Flatten(input_shape=(28,28)),
        keras.layers.Dense(128,activation=tf.nn.relu),
        keras.layers.Dense(10,activation=tf.nn.softmax)
        ])
#Flatten görüntüyü 28*28 boyutun da tek boyutlu sayı dizisine cevirir
#Modeli derleyelim
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#Artık modelimizi eğitebiliriz
model.fit(traim_images,train_labels,epochs=3)
#%% ARTIK MODELİMİZİ TEST EDEBİLİRİZ
test_loss,test_acc=model.evaluate(test_images,test_labels)
print(test_acc)#Burada dogruluk oranını verir
#Bende 0.86 oranında dogru cıktı

#%% Şimdi tahminleme yapalım
pre=model.predict(test_images)
print(pre[0])
#Burada 10 sayıdan oluşan dizi elde ederiz yani 10 degerin dogruluk oranınıdır
#Hangisi en fazlaysa sonuç odur
print(np.argmax(pre[0]))#en büyük hangisi bakalım 
#Sonuc 9 verdi yani 9. kıyafet
print(test_labels[0])#Buda bizim gerçek sonuçumuz
#Burda da sonuc 9 cıktı yani modelimiz dogru tahmin etmiş 
#%%
#Hadi 10 farklı sonucun tamamına bakalım 
def plot_image(i,pre_array,true_labels,image):
    pre_array,true_labels,image=pre_array[i],true_labels[i],image[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image,cmap=plt.cm.binary)
    pre_label=np.argmax(pre_array)
    if  pre_label==true_labels:
        color="blue"
    else:
        color="red"
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[pre_label],
                                100*np.max(pre_array),
                                class_names[true_labels]),
                                color=color)
def plot_value_array(i, pre_array, true_label):
    pre_array, true_label = pre_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), pre_array, color="#777777")
    plt.ylim([0, 1]) 
    predicted_label = np.argmax(pre_array)
     
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
#Evet fonk yazdık şimdi kullanalım
    
#0.ıncı image bakalım
i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i,pre,test_labels,test_images)
plt.subplot(1,2,2)
plot_value_array(i,pre,test_labels)
plt.show()
#Evet ben de güzel bir görsel oldu 
#%% Hadi biraz daha görselleştirelim
#Dogru olma olasığı mavi yanlış olma olasıgı kırmızı olacak 
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, pre, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, pre, test_labels)
plt.show()
#%% Tek resim üzerinde tahminleyelim
img=test_images[0]
img=(np.expand_dims(img,0))
pre_single=model.predict(img)
plot_value_array(0, pre_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)
plt.show()



    
    
    
    
    
    
    
    
    
    
    





































    
    
    
    
    
    


























