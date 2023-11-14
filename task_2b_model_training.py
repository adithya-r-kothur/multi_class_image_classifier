# In[1]:
import os
import matplotlib.pyplot as plt

# In[1]:

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Dropout, Flatten, Dense, BatchNormalization

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# In[2]:

from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet101, preprocess_input

# In[3]:


# In[4]:
# os.chdir('Downloads/task_2b_evaluator-20231020T113830Z-001/task_2b_evaluator/')

img_width=256; img_height=256
batch_size=16

# In[5]:
TRAINING_DIR = 'C:/Users/Adithya r kothur/Downloads/task_2b_evaluator-20231020T113830Z-001/task_2b_evaluator/task_2b_data/train2'
TRAINING_DIR = TRAINING_DIR.strip()
# data_dir = Path("task_2b_data")
# image_path = data_dir
# # image_path = data_dir / "test"
# if image_path.is_dir():
#     print(f"{image_path} directory exists.")
# else:
#     print(f"Did not find {image_path} directory, creating one...")
#     image_path.mkdir(parents=True, exist_ok=True)


# TRAINING_DIR = image_path / "train"
# VALIDATION_DIR = image_path / "test"

train_datagen = ImageDataGenerator(rescale = 1/255.0)

# In[6]:

train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    target_size=(img_height, img_width)
                                                    )

# In[7]:

VALIDATION_DIR = 'C:/Users/Adithya r kothur/Downloads/task_2b_evaluator-20231020T113830Z-001/task_2b_evaluator/task_2b_data/train2'
# VALIDATION_DIR = VALIDATION_DIR.strip()

validation_datagen = ImageDataGenerator(rescale = 1/255.0)

# In[8]:

validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              target_size=(img_height, img_width)
                                                             )

# In[9]:
callbacks = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')        
# autosave best Model
best_model_file = '.../resnet101_drop_batch_best_weights_256.h5'
best_model = ModelCheckpoint(best_model_file, monitor='val_acc', verbose = 1, save_best_only = True)

# In[9]:

wp = '.../resnet101_weights_tf_dim_ordering_tf_kernels_notop.h5'
resnet101_base = ResNet101(include_top=False,
                           input_tensor=None, input_shape=(img_height, img_width,3))

# In[10]:

print('Adding new layers...')
output = resnet101_base.get_layer(index = -1).output  
output = Flatten()(output)
# let's add a fully-connected layer
output = Dense(512,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
output = Dense(512,activation = "relu")(output)
output = BatchNormalization()(output)
output = Dropout(0.2)(output)
# and a logistic layer -- let's say we have 4 classes
output = Dense(5, activation='softmax')(output)
print('New layers added!')

# In[11]:

resnet101_model = Model(resnet101_base.input, output)
for layer in resnet101_model.layers[:-7]:
    layer.trainable = False

resnet101_model.summary()

# In[12]:

resnet101_model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics =['accuracy'])

# In[13]:

history = resnet101_model.fit_generator(train_generator,
                              epochs=10,
                              verbose=1,
                              validation_data=validation_generator,
                              callbacks = [callbacks, best_model]
                              )


resnet101_model.save('resnet101_model.h5')
resnet101_model.save_weights('resnet101_weights.h5')