import os
import sys
import json

sys.path.insert(0, 'src')

from data.etl import get_content_and_cites
from data.etl import get_data
from models.train_model import training
from models.predict_model import predictions

def main():
    path1 = "data/raw/citeseer.cites"
    path2 = "data/raw/citeseer.content"

    citeseer_content_feats, citeseer_cites = get_content_and_cites(path1, path2)

    # add to out now 
    get_data(citeseer_content_feats, citeseer_cites, "data/out/")

    model, gen, generator, nodes, target_encoding = training(citeseer_content_feats, citeseer_cites)

    return predictions(model, gen, generator, nodes, target_encoding)
    
if __name__ == '__main__':
    main()