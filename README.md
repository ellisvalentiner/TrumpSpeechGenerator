# Trump Speech Generator

This is a LSTM model to generate text based on speeches by President Trump.

The speeches are sourced from [hrbrmstr/orangetext](https://github.com/hrbrmstr/orangetext).

Since the text generating model is a LSTM, and RNNs are slow, it takes some time before the model begins to generate output that isn't complete gibberish.

```bash
python src.py
```
