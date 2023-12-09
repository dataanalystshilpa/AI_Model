#!/usr/bin/env python
# coding: utf-8

# In[2]:


pip install mysql-connector-python


# In[ ]:


import torch  # Or any libraries needed for your model
from flask import Flask, request, jsonify
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# Load the sentiment analysis model
f = torch.load('sentiment_model.pt')  # Load your model here


# In[4]:


app = Flask(__name__)

@app.route('/predict_sentiment', methods=['GET','POST'])
def predict_sentiment():
    data = request.json
    text = data.get('text')

    if not text:
        return jsonify({'error': 'Text not provided'}), 400

    # Make prediction using your model
    prediction = model.predict(text)  # Replace this line with your model prediction logic

    # Logging
    session = Session()
    log_entry = Log(text=text, prediction=prediction)
    session.add(log_entry)
    session.commit()
    session.close()

    return jsonify({'text': text, 'sentiment_prediction': prediction}), 200

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')


# In[ ]:


from flask import Flask, request
import mysql.connector

app = Flask(__name__)
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="shilpa1234567890",
    database="python"
)

try:
    conn = mysql.connector.connect(host=host, user=user, password=password, database=database)
    if conn.is_connected():
        print('Connected to MySQL database')
        
    # Perform database operations here
    
    conn.close()  # Close the connection when done
    
except mysql.connector.Error as e:
    print(f"Error connecting to MySQL: {e}")

#torch.save(f.state_dict(), 'sentiment_model.pt')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




