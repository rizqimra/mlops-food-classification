from train import test_loader
import tensorflow as tf

def evaluate_model():
    # Load trained model
    model = tf.keras.models.load_model("./models/best_model.keras")

    # Evaluate model
    eval_result = model.evaluate(test_loader)
    print(f"Test Loss: {eval_result[0]}, Test Accuracy: {eval_result[1]}")

if __name__ == "__main__":
    evaluate_model()
