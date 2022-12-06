import pandas as pd
import os
import numpy as np
from stellargraph import StellarGraph
import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GCN
from tensorflow.keras import layers, optimizers, losses, metrics, Model
from tensorflow.keras.callbacks import EarlyStopping
from sklearn import preprocessing, model_selection
from IPython.display import display, HTML

def predictions(model, test_gen, generator, nodes, target_encoding):
    test_metrics = model.evaluate(test_gen)
    all_nodes = nodes.index
    all_gen = generator.flow(all_nodes)
    all_predictions = model.predict(all_gen)
    node_predictions = target_encoding.inverse_transform(all_predictions.squeeze())
    
    df = pd.DataFrame({"Predicted": node_predictions, "Actual": nodes})
    accuracy = df.loc[df['Predicted'] == df['Actual']].shape[0] / df.shape[0]
    print(df)
    print("The classification accuracy of this GCN on CiteSeer data is: " + str(accuracy))
    return df, accuracy