import tensorflow as tf
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
from model import get_model

from flask import Flask, request, jsonify
app = Flask(__name__)


model, _ = get_model()
model.load_weights('assets/ArcFaceModel_best.h5')
embed_model = tf.keras.models.Model(inputs=model.input[0], 
                    outputs=model.get_layer('tf.math.l2_normalize').output)
labse_model = SentenceTransformer('assets/labse_model')
knn = pickle.load(open('assets/knnpickle', 'rb'))
reference_id_db = np.load('assets/reference_id_db.npy', allow_pickle=True)


@app.route('/match_products/', methods=['GET', 'POST'])
def inference():
    
    data = request.json
    
    labse_arc_embeddings = []
    for d in data:
        labse_embedding = np.zeros((17, 768))
        corpus = [d['name']]
        for prop in d['props']:
            corpus.append(prop.replace('\t', ' '))
        corpus_embedding = np.array(labse_model.encode(corpus))
        labse_embedding[:corpus_embedding.shape[0], :] = corpus_embedding
        labse_arc_embeddings.append(labse_embedding)
    labse_arc_embeddings = np.array(labse_arc_embeddings)

    arc_embeddings = embed_model.predict(labse_arc_embeddings)

    dist, indx = knn.kneighbors(arc_embeddings, return_distance=True)

    reference_ids = reference_id_db[indx[:, 0]]

    res = []
    for d, refid in zip(data, reference_ids):
        res.append({"id": d['id'], "reference_id": refid})

    return jsonify(res)
  
  
if __name__ == '__main__':
    app.run()
