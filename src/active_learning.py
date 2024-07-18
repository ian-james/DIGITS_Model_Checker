import os
import tensorflow as tf
assert tf.__version__.startswith('2')

from mediapipe_model_maker import image_classifier

import matplotlib.pyplot as plt


# Download the flower dataset
image_path = tf.keras.utils.get_file(
    'flower_photos.tgz',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

print(image_path)
labels = []
for i in os.listdir(image_path):
  if os.path.isdir(os.path.join(image_path, i)):
    labels.append(i)
print(labels)



# Review the datasets
NUM_EXAMPLES = 5

for label in labels:
  label_dir = os.path.join(image_path, label)
  example_filenames = os.listdir(label_dir)[:NUM_EXAMPLES]
  fig, axs = plt.subplots(1, NUM_EXAMPLES, figsize=(10,2))
  for i in range(NUM_EXAMPLES):
    axs[i].imshow(plt.imread(os.path.join(label_dir, example_filenames[i])))
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
  fig.suptitle(f'Showing {NUM_EXAMPLES} examples for {label}')

plt.show()

# Setup the model
spec = image_classifier.SupportedModels.MOBILENET_V2
hparams = image_classifier.HParams(export_dir="exported_model")
options = image_classifier.ImageClassifierOptions(supported_model=spec, hparams=hparams)

# SEtup Data
data = image_classifier.Dataset.from_folder(image_path)
train_data, remaining_data = data.split(0.8)
test_data, validation_data = remaining_data.split(0.5)


model = image_classifier.ImageClassifier.create(
    train_data = train_data,
    validation_data = validation_data,
    options=options,
)

# Evaluate Model
loss, acc = model.evaluate(test_data)
print(f'Test loss:{loss}, Test accuracy:{acc}')

model.export_model()

#Tuning the model
hparams=image_classifier.HParams(epochs=15, export_dir="exported_model_2")
options = image_classifier.ImageClassifierOptions(supported_model=spec, hparams=hparams)
options.model_options = image_classifier.ModelOptions(dropout_rate = 0.07)
model_2 = image_classifier.ImageClassifier.create(
    train_data = train_data,
    validation_data = validation_data,
    options=options,
)

loss, accuracy = model_2.evaluate(test_data)
print(f'Test loss:{loss}, Test accuracy:{accuracy}')


# Quantization the model for deployment
from mediapipe_model_maker import quantization
quantization_config = quantization.QuantizationConfig.for_int8(train_data)

model.export_model(model_name="model_int8.tflite", quantization_config=quantization_config)


