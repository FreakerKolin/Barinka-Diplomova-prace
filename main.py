import os
from src.train import train_model
from src.apply import apply_model

if __name__ == "__main__":
    model_path = "models/cnn_model.pth"

    # jen když model ještě neexistuje, spustíme trénink
    if not os.path.exists(model_path):
        train_model(
            input_dir="data/input_wavs",
            target_dir="data/target_wavs",
            save_path=model_path,
            epochs=5
        )
    else:
        print("Model už existuje, netrénuju znovu.")

    # vždycky aplikace na test
    apply_model(
        model_path=model_path,
        input_wav="data/test_wavs/test.wav",
        output_wav="data/test_wavs/test_processed.wav"
    )