import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# mixed precision
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy("mixed_float16")

# Paths dataset
base_dir = "Indonesian-Food-1"
train_dir = f"{base_dir}/train"
valid_dir = f"{base_dir}/valid"
test_dir = f"{base_dir}/test"

# Data generators
data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

data_gen_valid = ImageDataGenerator(rescale=1.0 / 255)

train_loader = data_gen.flow_from_directory(
    train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

valid_loader = data_gen_valid.flow_from_directory(
    valid_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

test_loader = data_gen_valid.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical"
)

# Load MobileNet
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dense(train_loader.num_classes, activation="softmax", dtype="float32")(
    x
)  # Use float32 for final layer

model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Early stopping
checkpoint = ModelCheckpoint(
    "best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1
)
early_stop = EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)

# Train
history = model.fit(
    train_loader,
    validation_data=valid_loader,
    epochs=20,
    callbacks=[checkpoint, early_stop],
    verbose=1,
)

# Evaluate model
eval_result = model.evaluate(test_loader)
print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")
