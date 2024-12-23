from train import test_loader

# Evaluate model
model = model = tf.keras.models.load_model("best_model.keras")

eval_result = model.evaluate(test_loader)
print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")