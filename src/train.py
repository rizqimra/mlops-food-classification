import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy("mixed_float16")

# Dataset paths
base_dir = "data"
train_dir = f"{base_dir}/train"
valid_dir = f"{base_dir}/valid"
test_dir = f"{base_dir}/test"

# Data generators
data_gen = ImageDataGenerator(rescale=1.0 / 255, rotation_range=20, width_shift_range=0.2,
                              height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

data_gen_valid = ImageDataGenerator(rescale=1.0 / 255)

train_loader = data_gen.flow_from_directory(train_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")
valid_loader = data_gen_valid.flow_from_directory(valid_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")
test_loader = data_gen_valid.flow_from_directory(test_dir, target_size=(224, 224), batch_size=32, class_mode="categorical")

# MobileNet base model
base_model = MobileNet(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dense(train_loader.num_classes, activation="softmax", dtype="float32")(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

if __name__ == "__main__":
	# MLflow experiment tracking
	mlflow.start_run()
	
	# Log hyperparameters
	mlflow.log_param("model", "MobileNet")
	mlflow.log_param("optimizer", "Adam")
	mlflow.log_param("learning_rate", 0.001)
	mlflow.log_param("epochs", 20)
	mlflow.log_param("batch_size", 32)
	
	# Early stopping and model checkpointing
	checkpoint = ModelCheckpoint("best_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
	early_stop = EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
		
	# Train model
	history = model.fit(train_loader, validation_data=valid_loader, epochs=20,
	                    callbacks=[checkpoint, early_stop], verbose=1)
	
	# Log metrics
	mlflow.log_metric("final_train_accuracy", history.history['accuracy'][-1])
	mlflow.log_metric("final_val_accuracy", history.history['val_accuracy'][-1])
	
	# Log model
	mlflow.tensorflow.log_model(model, "model")
	
	# End the MLflow run
	mlflow.end_run()
