import re
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, Form
from starlette.response import HTMLResponse
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

@app.get('/predict', response_class=HTMLResponse)
def take_input():
    return """
        <form method='post'>
        <input maxlength='28' name='text' type='text' value='Text emotion to be tested'/>
        <input type='submit'/>
    """
