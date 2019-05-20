# -*- coding: utf-8 -*-
"""
Created on Mon May 20 23:04:29 2019

@author: Salim
"""

import keras 
import numpy as np
"""
inputs      outputs
A     B      C=A*B
0     0       0
0     1       0
1     0       0
1     1       1

Bu girdi ve sonucları ysa modelimize yaptırmaya çalışalım
"""
inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype = float)
outputs = np.array([[0.0], [0.0], [0.0], [1.0]], dtype = float)

#Model oluşturalım
#Modelimize girdi katmanı ve cıkış katmanımız ekleyelim 
first_model=keras.Sequential()
first_model.add(keras.layers.Dense(units=2,input_shape=[2]))
first_model.add(keras.layers.Dense(units=1))
#Modelimizi compile edelim

first_model.compile(optimizer="sgd",loss="mean_squared_error",metrics=["accuracy"])

#Modelimizi fit edelim yani eğitelim 
first_model.fit(inputs,outputs,epochs=300,verbose=0)


#Modelimizi çalıştırıp bir şonuc seçelim
"""print(first_model.predict(np.array([[0.0,0.0]])))
print(first_model.predict(np.array([[1.0,1.0]])))"""
#Burda bir sonuç aldık ama bizim için tam birşey ifade etmiyor 
#Şimdi tüm girdileri np dizisi şeklinde yollayalım ve sonuçları görelim
zero_zero=np.array([[0.0,0.0]])
zero_one=np.array([[0.0,1.0]])
one_zero=np.array([[1.0,0.0]])
one_one=np.array([[1.0,1.0]])
pre_inputs=[zero_zero,zero_one,one_zero,one_one]
"""for i in pre_inputs:
    print("****")
    print(first_model.predict(i))"""
#Şimdi hepsini birlikte gördük ama yine tam olarak birşey ifade etmiyor 
#Şonucları inceledigimiz zama 3 tanesinin 0.5 küçük bir tanesinin büyük olduğu dikkanizi çekmiştir
#Ozaman eşik degeri verebiliriz
def Eşik_Degeri(inputs):
    x=first_model.predict(inputs)
    if x>0.5:
        print("{}:Bunun sonucu 1".format(x))
    else:
        print("{}:Bunun sonucu 0".format(x))
#Artık yukardaki for döngüsünde bu fonk kullanalım ve onu yorum satırı yapalım
#İlk başta ki sonuçları yorum satırı yapabiliriz
for i in pre_inputs:
    Eşik_Degeri(i)
#Artık işlemimiz tamam girdilerimize göre modelimiz bize sonuçları dogru şekilde veriyor
















