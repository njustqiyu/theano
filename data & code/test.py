import cPickle as pickle

"""
An example of how to load a trained model and use it
to predict labels.
"""

# load the saved model
classifier = pickle.load(open('best_model.pkl'))

# compile a predictor function
predict_model = theano.function(
	inputs=[classifier.input],
	outputs=classifier.y_pred)

# We can test it on some examples from test test
dataset='mnist.pkl.gz'
datasets = load_data(dataset)
test_set_x, test_set_y = datasets[2]
test_set_x = test_set_x.get_value()

predicted_values = predict_model(test_set_x[:10])
print("Predicted values for the first 10 examples in test set:")
print(predicted_values)
	

    