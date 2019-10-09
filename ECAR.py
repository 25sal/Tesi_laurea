import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from datetime import datetime
import time
import numpy as np
from sklearn.model_selection  import GridSearchCV
from sklearn import preprocessing
import argparse
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import MinMaxScaler

class ECAR:
    energia = []
    Day_hour = []
    Day_week = []
    Durata_ricarica = []
    Next_charge = []
    Weekend = []
    event = []
    en = [] #valori di energia assorbiti nel corso del tempo
    maxpower = []
    X = pd.DataFrame()
    y = pd.DataFrame()
    X_scaled = pd.DataFrame()
    y_scaled = pd.DataFrame()
    differenza = 0
    ymin = 0
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    def __init__(self,path):
        tree = ET.parse(path)
        root = tree.getroot()
        index = 0
        for i in root[:]:
            self.event.append(float(root[index][0].text))
            self.energia.append(float(root[index][1].text))
            self.Day_hour.append(float(root[index][2].text))
            self.Day_week.append(float(root[index][3].text))
            self.Durata_ricarica.append(float(root[index][4].text))
            self.en.append(float(root[index][6].text))
            self.maxpower.append(float(root[index][7].text))
            self.Next_charge.append(float(root[index][8].text))
            self.Weekend.append((root[index][5].text))
            index = index + 1
        df = pd.DataFrame({"Event" : self.event,"Energia_assorbita_Wh" : self.energia,"Day_hour" : self.Day_hour,"Day_week": self.Day_week,"Next_charge_minuti": self.Next_charge,"Weekend": self.Weekend,"Durata_ricarica_minuti": self.Durata_ricarica,"Energia" : self.en, "MaxPower" : self.maxpower})
        self.y = df[['Next_charge_minuti']]
        self.X = df[['Event','Energia_assorbita_Wh','Day_hour','Day_week','Durata_ricarica_minuti',"Energia","MaxPower"]]
        self.scaler1 = preprocessing.StandardScaler().fit(self.X)
        self.scaler2 = preprocessing.StandardScaler().fit(self.y)
        self.scale_data()

    def scale_data(self):
        self.X_scaled = self.scaler1.transform(self.X)
        self.y_scaled = self.scaler2.transform(self.y)
        
    
    def show_test_result(self,y_reale,y_pred):
            print('Mean Absolute Error:', metrics.mean_absolute_error(y_reale, y_pred))
            print('Mean Squared Error:', metrics.mean_squared_error(y_reale, y_pred))
            print('Root Mean Squared Error:', np.sqrt(mean_squared_error(y_reale,y_pred)))
            ascisse1 = []
            for i in (range(0,len(y_pred))):
                      ascisse1.append(i)
                    #  print("Inizio prossima ricarica (reale)",y_reale[i],"Inizio prossima ricarica (predetto)",y_pred[i])
                        
            plt.plot(ascisse1,[x/60 for x in y_reale] ,label = 'Reale')
            plt.plot(ascisse1,[x/60 for x in y_pred] ,label = 'Predizione')
            plt.legend(loc = 'upper left')
            plt.show()
            plt.figure(2)
    
    def show_result(self,y_pred,length):
            print("Prossima ricarica tra (minuti)")
            print(y_pred[0])
            for i in range (1, length):    #aspetto che l'utente mi dica next per mostrare la successiva predizione
                command = input()
                if (command == "next"):
                    print("Prossima ricarica tra (minuti)")
                    print(y_pred[i])
                else:
                    i = i - 1
        
    
    def test(self,train):
        self.scale_data()
        algoritmo = "dynamic"
        parser = argparse.ArgumentParser(description='Get the key')  #leggo l'algoritmo da utilizzare
        parser.add_argument('mood', action="store")
        algoritmo = parser.parse_args()
        
        svr_rbf = SVR(kernel = 'poly', gamma = 0.06, C = 120)  #set kernel
        size = int((len(self.X_scaled[:])/100)*train)   #calcolo il numero dei campioni su cui voglio fare il training iniziale
        
        if (algoritmo.mood == "dynamic"):
            X_train = self.X_scaled[:size,:]
            X_test = self.X_scaled[size,:]
            y_train = self.y_scaled[:size,:]
            y_test = self.y_scaled[size,:]
            y_pred = []
            y_reale = []
            for i in range(0,len(self.X_scaled[:]) - size - 1):    #effettuo la predizione ed effettuo nuovamente il training
                y_pred.append(abs(self.scaler2.inverse_transform(svr_rbf.fit(X_train, y_train.ravel()).predict(X_test.reshape(1,-1)))))     #effettuo la prediione dei risultati
                training = svr_rbf.fit(X_train,y_train.ravel())
                X_train = np.vstack([X_train,X_test])   #aggiungo una riga al modello di training
                X_test = self.X_scaled[size + i + 1]   #faccio il test sul campione successivo
                y_train = np.append(y_train,y_test)
                y_test = self.y_scaled[size + i +1]
                y_reale.append(self.scaler2.inverse_transform(y_test))
            del y_reale[0]
            y_reale.append(0)
            self.show_test_result(y_reale,y_pred)
        else:
            if (algoritmo.mood == "SVR"):
                 X_train, X_test, y_train, y_test = train_test_split(self.X_scaled,self.y_scaled ,test_size=1 - train/100, random_state=0,shuffle = False)
                 training = svr_rbf.fit(X_train,y_train.ravel())
                 y_pred = training.predict(X_test)
                 print(X_test)
                 self.show_test_result(self.scaler2.inverse_transform(y_test),abs(self.scaler2.inverse_transform(y_pred)))
    
    def training(self):
        svr_rbf = SVR(kernel = 'poly', gamma = 0.06, C = 120)
        model_training = svr_rbf.fit(self.X_scaled,self.y_scaled.ravel())
        dump(model_training,'/home/just/Documenti/training_model')
        return model_training
    
    def predict(self,prediction):
            df1 = pd.read_csv(prediction)
            training = load('/home/just/Documenti/training_model')
            parser = argparse.ArgumentParser(description='Get the key')  #leggo l'algoritmo da utilizzare
            parser.add_argument('mood', action="store")
            algoritmo = parser.parse_args()
            if (algoritmo.mood == "SVR"):                                   #leggo il mood e mostro i risultati predetti
                    y_pred = training.predict(self.scaler1.transform(df1.iloc[:,:7])).reshape(-1,1)
                    self.show_result(abs(y_pred),len(df1.iloc[:]) - 1)
                    
                
            if (algoritmo.mood == "dynamic"):   
                y_reale = []
                for i in range (0,len(df1.iloc[:,1])-1):
                    y_pred = training.predict(self.scaler1.transform(df1.iloc[i,:7].values.reshape(1,-1)))
                    self.X_scaled = np.vstack([self.X_scaled,self.scaler1.transform(df1.iloc[i,:7].values.reshape(1,-1))])
                    self.y_scaled = np.vstack([self.y_scaled,df1.iloc[i,8].reshape(1,1)])
                    training = self.training()
                    y_reale.append(y_pred)
                self.show_result([abs(y) for y in y_reale],len(df1.iloc[:]) - 1)

auto = ECAR('/home/just/Documenti/data.xml')
auto.test(85)
auto.training()
auto.predict("/home/just/Scaricati/prediction_model.csv")
                
            
        
    
     
    
