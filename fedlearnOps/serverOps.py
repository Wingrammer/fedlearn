import numpy as np

import json
import matplotlib.pyplot as plt

import tensorflow as tf
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.compat.v1.keras import backend as K

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)
    
    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value. 

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Tensor): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0

        for i in range(len(pos_weights)):
            # for each class, add average weighted loss for that class 
            loss += -(K.mean( pos_weights[i] * y_true[:,i] * K.log(y_pred[:,i] + epsilon) + \
                                neg_weights[i] * (1 - y_true[:,i]) * K.log(1 - y_pred[:,i] + epsilon), axis = 0))
        return loss
    
    return weighted_loss


class Flserver():
    """Server computations"""

    def __init__(self):

        super(Flserver, self).__init__()
        self.clients = np.array([])
        self.K = 0                                  
        self.C = 7/10
        self.current_weights = []
        self.n_round = 1
        self.max_round = 4
        self.scores = np.array([])

    def json_serialize(self, weights):
        serialized_weights = lambda a: [i.tolist() for i in a]
        return serialized_weights(weights)

    def json_desrialize(self, weights):
        deserialized_weights = lambda a: [np.array(i) for i in a]
        return deserialized_weights(weights)
        
    def build_model(self, pos_weights, neg_weights):

        base_model = DenseNet121(include_top=False)
        x = base_model.output

        # add a global spatial average pooling layer
        x = GlobalAveragePooling2D()(x)

        # and a logistic layer
        predictions = Dense(2, activation="sigmoid")(x)

        model = Model(inputs=base_model.input, outputs=predictions)
        model.compile(optimizer=Adam(0.0001), loss=get_weighted_loss(pos_weights, neg_weights), metrics=[tf.keras.metrics.BinaryAccuracy()])

        return self.json_serialize(model.get_weights())

    def on_connect(self, sid, environ):
        
        print(f"Client connected: {sid}")

    def on_model_request(self, sid, data):
        print("server preparing the model")
        pos_weights = msgpack.unpackb(data["pos"], object_hook=m.decode)
        neg_weights = msgpack.unpackb(data["neg"], object_hook=m.decode)
        self.model = self.build_model(pos_weights, neg_weights)
        
        print(self.model.summary())

        server_packet = {"model": self.model.to_json(), "n_round": 1}

        return server_packet


    def on_disconnect(self, sid):
        
        print(f"Client disconnected: {sid}")

    def on_complete_round(self, data):
        
        print("Round completed")

        self.clients = np.array([a for a in data])
        self.K = self.clients.shape[0]
        m = max(self.C*self.K, 1)
        self.clients_updaters = np.random.choice(self.clients, int(m))
        n_k = np.array([i['n_train'] for i in self.clients_updaters])
        n_k = n_k.reshape(n_k.shape[0], 1)
        n_k_test = np.array([i['n_test'] for i in self.clients_updaters])
        n_test = np.sum(n_k_test)
        n = np.sum(n_k)
        current_weights = np.array([self.json_desrialize(i['weights']) for i in self.clients_updaters])
        current_scores = np.array([i['scores'] for i in self.clients_updaters])
        new_scores = n_k_test*np.sum(current_scores, axis = 0)/n_test
        new_weights = np.sum(n_k*current_weights/n, axis = 0)
        self.current_weights = new_weights
        self.scores = np.append(self.scores, new_scores)
        
        
        self.n_round = self.n_round + 1
        server_packet = {"model": self.json_serialize(new_weights.tolist()), "n_round": self.n_round}
        return server_packet

        """ else:

            fig, axs = plt.subplots(2, 1)
            axs[0].plot([i+1 for i in range(self.max_round)], self.scores[:,0])
            axs[0].set_xlabel('Round')
            axs[0].set_ylabel('Agg Test Score')

            axs[1].plot([i+1 for i in range(self.max_round)], self.scores[:,1])
            axs[1].set_xlabel('Round')
            axs[1].set_ylabel('Agg Test Accuracy')

            fig.tight_layout()
            plt.show()

            print("model updated") """

        

