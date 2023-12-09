#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('writefile', 'Dockerfile', '\n# Use a base image with the desired environment (e.g., Python)\nFROM python:3.9\n\n# Set the working directory inside the container\nWORKDIR /app\n\n# Copy the necessary files into the container (e.g., model files, code)\nCOPY requirements.txt /app/requirements.txt\nCOPY your_model.py /app/your_model.py\nCOPY web_service.py /app/web_service.py\n\n# Install dependencies\nRUN pip install --no-cache-dir -r requirements.txt\n\n# Expose the port where the web service will run (change port number if needed)\nEXPOSE 8080\n\n# Define the command to start the web service (modify according to your application)\nCMD ["python", "web_service.py"]\n')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




