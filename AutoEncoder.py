from sklearn.base import BaseEstimator, TransformerMixin
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.decomposition import PCA
from keras.layers import Input, Dense
from keras.models import Model


class AutoEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, layers=None, epochs=10):
        self.layers = layers
        self.epochs = epochs
        self.encoder = None

    def fit(self, X, y=None):
        input_dims = Input(shape=(180,))
        encoded = Dense(self.layers[0], activation='relu')(input_dims)
        encoded = Dense(self.layers[1], activation='relu')(encoded)
        decoded = Dense(self.layers[2], activation='relu')(encoded)
        decoded = Dense(180, activation='sigmoid')(decoded)

        autoencoder = Model(input_dims, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        autoencoder.fit(X, X,
                        epochs=self.epochs,
                        batch_size=50,
                        shuffle=True, verbose=0)
        self.encoder = Model(input_dims, encoded)
        return self

    def transform(self, X, y=None):
        return self.encoder.predict(X)

    def get_params(self, deep=True):
        return {"layers": self.layers,
                "epochs": self.epochs}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
