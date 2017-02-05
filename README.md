# Trump Speech Generator

This is a LSTM model to generate text based on speeches by President Trump.

The speeches are sourced from [hrbrmstr/orangetext](https://github.com/hrbrmstr/orangetext).

Since the text generating model is a LSTM, and RNNs are slow, it takes some time before the model begins to generate output that isn't complete gibberish.

I suggest installing the requirements in a virtual environment:

```bash
$ git clone git@github.com:ellisvalentiner/TrumpSpeechGenerator.git
$ cd TrumpSpeechGenerator
TrumpSpeechGnerator $ mkvirtualenv Keras
(Keras) TrumpSpeechGnerator $ pip install -r requirements.txt
(Keras) TrumpSpeechGenerator $ ipython src.py > log
```

You then `tail -f log` to check in on the model:

```python
Using TensorFlow backend.
corpus length: 163283
total chars: 92
nb sequences: 54408
Vectorization...
Build model...

--------------------------------------------------
Iteration 1
Epoch 1/10
54400/54408 [============================>.] - ETA: 0s - loss: 2.7732Epoch 00000: loss improved from inf to 2.77315, saving model to models/weights-00-2.7731.hdf5
54408/54408 [==============================] - 1646s - loss: 2.7731
```

Notice it took 1646s to complete one epoch on my MacBook Pro (Retina, 15-inch, Mid 2014) with the 2.5 GHz Intel Core i7 processor.
