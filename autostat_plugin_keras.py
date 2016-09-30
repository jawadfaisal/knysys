
if o.engine != "Keras": 
    # need better register_engine
    throw

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.layers import *
from keras.models import model_from_json

from sklearn.cross_validation import train_test_split


class Autostat_Keras(Autostat):
    def __init__(self, o=None):
        if o: super(Autostat_Keras, self).__init__(o=o)
        else: super(Autostat_Keras, self).__init__()

    def set_model(self):
        super(Autostat_Keras, self).set_model()
        y = np.array(self.df[self.o.target])
        X = np.array(self.df.drop(self.o.target,axis=1))
        
        dims = len(self.df.columns)-1
        
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(X, y, train_size=0.5, random_state=0)

        model = Sequential()
        model.add(Dense(100, input_dim=dims, init='normal', activation='relu'))
        model.add(Dense(100, input_dim=dims, init='normal',activation='relu'))
        model.add(Dense(150, input_dim=dims, init='normal',activation='softplus'))
        model.add(Dense(150, input_dim=dims, init='normal',activation='softplus'))
        model.add(Dense(1, init='normal', activation='sigmoid'))
        model.compile(loss='msle', optimizer='adam', metrics=['accuracy'])
        self.model=model

    def train(self):
        #super(Autostat_Keras, self).train() #suppress this
        self.model.fit(self.train_X, self.train_y, verbose=1, batch_size=1, nb_epoch=5)
        
    def save_model(self):
        self.disable_model_pickle = True
        super(Autostat_Keras, self).save_model()
        json_string = self.model.to_json()
        open('my_model_architecture.json', 'w').write(json_string)
        self.model.save_weights('my_model_weights.h5', overwrite=True)
        
    def load_model(self):
        super(Autostat_Keras, self).load_model()
        self.model = model_from_json(open('my_model_architecture.json').read())
        self.model.load_weights('my_model_weights.h5')
        self.model.compile(loss='msle', optimizer='adam', metrics=['accuracy'])

register_engine(Autostat_Keras)


    