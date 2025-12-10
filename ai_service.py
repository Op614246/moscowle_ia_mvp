import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from joblib import load, dump
from sklearn.svm import SVC

MODEL_PATH= 'ai_models/svm_model.pkl'

def train_model():
    X=[]
    Y=[]
    for _ in range (500):
        acc = np.random.uniform(0, 100)
        time = np.random.uniform(500, 3000)
        
        if acc < 80 and time <1000: label = 1
        elif acc < 50 and time >2000: label = 2
        else: label = 0

        X.append([acc, time])
        Y.append(label)

    model = SVC(kernel='rbf', probability=True)
    model.fit(X, Y)
    # ensure the directory for the model exists
    model_dir = os.path.dirname(MODEL_PATH)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    dump(model, MODEL_PATH)
    print("Modelo entrenado y guardado")

def predict_level(accuracy, avg_time):
    if not os.path.exists(MODEL_PATH):
        train_model()
    
    model = load(MODEL_PATH)
    # predict returns an array; take the first (and only) element
    pred = model.predict([[accuracy, avg_time]])[0]

    labels = {0: "Mantener Nivel", 1: "Avanzar Nivel", 2: "Retroceder/Apoyo"}
    
    return int(pred), labels[int(pred)]

def get_cluster(metrics_data):
    if len(metrics_data) < 3: return []
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(metrics_data)
    return kmeans.labels_

