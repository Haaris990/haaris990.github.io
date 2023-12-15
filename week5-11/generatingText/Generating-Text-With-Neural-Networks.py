#!/usr/bin/env python
# coding: utf-8

# # Generating Text with Neural Networks
# 

# Tensorflow is a Python library that this code utilises to create Shakespearean-style text using neural networks. It illustrates how, in addition to general generative AI, humanities are included in Large Language Models (LLM). Effective resources are necessary for machine learning projects with specialised datasets, but obtaining them might be difficult. My own computer's capacity restrictions forced me to lower the training data in order to avoid crashes. However, this choice produced some absurd and confusing results. The study emphasises how challenging it can be to carry out humanities initiatives with little funding. I included comments in my code to clarify some complex aspects.

# # Getting the Data

# In[1]:


import tensorflow as tf

shakespeare_url = "https://homl.info/shakespeare"  # shortcut URL
filepath = tf.keras.utils.get_file("shakespeare.txt", shakespeare_url)
with open(filepath) as f:
    shakespeare_text = f.read()


# In[2]:


print(shakespeare_text[:80]) # not relevant to machine learning but relevant to exploring the data


# # Preparing the Data

# The code allows us to analyse a dataset filled with shakespeares works. This computational analysis helps us recognise and understand different patterns in the text. This code firstly sets trains the model. Then is uses input validation to see if it has been trained correctly. Finally, tests how acurately it delivers an output. The primary aim is to train this model to create text like William Shakespeare. Training the model was a time consuming task. Therefore I lowered the ammount of training data. The sequence legnth is set to 100 and using tensorflow a random seed is initialised. The validation is set between 125,000 and 132,500. The validation is to see if the model has been trained correctly. The test data is set from 132,500 onwards. The test data evaluates its application of the data the model has learned from training. The purpose of these sets is to familiarise a computer program with modified Shakespearean text, aiding it in mastering the art of writing in the style of Shakespeare.Preparing the data may be time-consuming, but it's essential as it contributes to the computer's learning process. This highlights the significance of being mindful of the information we provide to the program.

# In[3]:


text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                   standardize="lower")
text_vec_layer.adapt([shakespeare_text])
encoded = text_vec_layer([shakespeare_text])[0]


# In[4]:


print(text_vec_layer([shakespeare_text]))


# In[5]:


encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars = 39
dataset_size = len(encoded)  # total number of chars = 1,115,394


# In[6]:


print(n_tokens, dataset_size)


# In[7]:


def to_dataset(sequence, length, shuffle=False, seed=None, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(sequence)
    ds = ds.window(length + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window_ds: window_ds.batch(length + 1))
    if shuffle:
        ds = ds.shuffle(100_000, seed=seed)
    ds = ds.batch(batch_size)
    return ds.map(lambda window: (window[:, :-1], window[:, 1:])).prefetch(1)


# In[11]:


length = 100
tf.random.set_seed(42)
train_set = to_dataset(encoded[:125_00], length=length, shuffle=True,
                       seed=42)
valid_set = to_dataset(encoded[125_000:132_500], length=length)
test_set = to_dataset(encoded[132_500:], length=length)


# # Building and Training the Model

# This code employs TensorFlow's Keras API to construct and train a neural network for language modeling, particularly on Shakespearean text. It begins by setting a random seed for TensorFlow to ensure reproducibility. he training process occurs over 10 epochs using specified training and validation datasets. The resulting model, once trained, can be useful for generating coherent and contextually relevant text in the style of Shakespeare.

# In[12]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    "my_shakespeare_model", monitor="val_accuracy", save_best_only=True)
history = model.fit(train_set, validation_data=valid_set, epochs=10,
                    callbacks=[model_ckpt])


# In[13]:


shakespeare_model = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Lambda(lambda X: X - 2),  # no <PAD> or <UNK> tokens
    model
])


# # Generating Text

# This code uses a trained language model (shakespeare_model) to generate text continuation based on an initial input. The input text, "To be or not to b," is fed into the model to predict the probabilities of the next word. The resulting word is a prediction for the next word in the sequence, enabling the model to generate coherent and contextually relevant text in a Shakespearean style. This process demonstrates the practical application of the trained language model for text generation tasks. As it you can see it finishes off the word be by adding the e character which is a Shakespeare quote.

# In[14]:


y_proba = shakespeare_model.predict(["To be or not to b"])[0, -1]
y_pred = tf.argmax(y_proba)  # choose the most probable character ID
text_vec_layer.get_vocabulary()[y_pred + 2]


# In[15]:


log_probas = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)  # draw 8 samples


# In[16]:


def next_char(text, temperature=1):
    y_proba = shakespeare_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]


# In[17]:


def extend_text(text, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char(text, temperature)
    return text


# In[18]:


tf.random.set_seed(42)  # extra code – ensures reproducibility on CPU


# The code below prints the result of the extend_text function applied to the input text "To be or not to be" with a temperature setting of 0.01. The output, "To be or not to be spate, who desires most that which would increase," reflects the continuation of the input text generated by the trained language model.

# In[19]:


print(extend_text("To be or not to be", temperature=0.01))


# In[20]:


print(extend_text("To be or not to be", temperature=1))


# In[21]:


print(extend_text("To be or not to be", temperature=100))


# The output, "To be or not to beg ,mt'&o3g:auy-$ wh-nse!pws&ertj-vberdq,!f-yje,znq," reflects the continuation of the input text generated by the trained language model. The high temperature setting introduces more randomness in character selection, leading to a more diverse and exploratory output. In this case, the generated text appears less constrained by deterministic patterns, resulting in a sequence that may include more unusual characters and combinations. 

# In[ ]:




