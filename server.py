from flask import Flask, jsonify, request
import pusher
from pymongo import MongoClient
from fedlearnOps.serverOps import Flserver
import numpy as np

flserver = Flserver()

pusher_client = pusher.Pusher(
  app_id='1143384',
  key='8ca34b7ce8e0e9ea5c45',
  secret='1b005fd2b7707fb41fb0',
  cluster='ap2',
  ssl=True
)

#pusher_client.trigger('my-channel', 'my-event', {'message': 'hello world'})
app = Flask(__name__)

connection_string = "mongodb+srv://admin:kkxdGerlCAf7cKqq@cluster0.ancln.mongodb.net/fedweights?retryWrites=true&w=majority"
client = MongoClient(connection_string)
db = client.fedweights

@app.route('/')
def home():
  
    return "Welcome"

@app.route('/initialize', methods=['GET'])
def initialize():
    pos_weights = request.args.get('pos_weights')
    neg_weights = request.args.get('neg_weights')
    weights = flserver.build_model(pos_weights, neg_weights)
    
    return jsonify({"weights": weights, "n_round": 1})

@app.route('/update', methods=['POST'])
def update():
    
    client_weights = db.local_weights
    server_weights = db.global_weights
    new_client = request.get_json()
    print(new_client)
    new_client_doc = client_weights.insert_one(new_client)
    this_round_clients = client_weights.find({"n_round": new_client["n_round"]})
    new_global_model = flserver.on_complete_round(this_round_clients)
    new_server_doc = server_weights.insert_one(new_global_model)

    return new_global_model


if __name__ == "__main__":
    
    app.run(debug=True)
    
 