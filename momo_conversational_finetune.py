import os
import dotenv

from utils.model_utils import initialize_model, setup_peft_model
from utils.trainer import ModelTrainer

dotenv.load_dotenv()


def main():
    # Initialize model
    model, tokenizer = initialize_model()
    model = setup_peft_model(model)

    # Create save directory
    save_path = "./mtn_momo_model"
    os.makedirs(save_path, exist_ok=True)

    # Train model
    trainer = ModelTrainer(model, tokenizer)
    trainer_instance = trainer.setup_trainer()
    trainer_instance.train()

    # Save the trained model
    # Push to hub
    model.push_to_hub_gguf(
        "JulienNyambal/mtn_momo_bot",
        tokenizer,
        token=os.getenv("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"),
    )

    # Remove any disk_offload calls here


if __name__ == "__main__":
    main()
