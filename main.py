'''
Questions Part A
'''

# Imports
import json
import argparse
import functools
from pprint import pprint

from Model import *

# Main Functions
# Wandb Sweep Function
def Model_Sweep_Run(wandb_data):
    '''
    Model Sweep Runner
    '''
    # Init
    wandb.init()

    # Get Run Config
    config = wandb.config
    N_EPOCHS = config.n_epochs
    BATCH_SIZE = config.batch_size

    ENCODER = config.encoder
    DECODER = config.decoder
    ENCODER_EMBEDDING_SIZE = config.encoder_embedding_size
    DECODER_EMBEDDING_SIZE = config.decoder_embedding_size
    ENCODER_N_UNITS = config.encoder_n_units
    DECODER_N_UNITS = config.decoder_n_units
    ACT_FUNC = config.act_func
    DROPOUT = config.dropout
    USE_ATTENTION = False

    print("RUN CONFIG:")
    pprint(config)

    # Get Inputs
    inputs = {
        "model": {
            "blocks": {
                "encoder": [
                    functools.partial(BLOCKS_ENCODER[ENCODER], 
                        n_units=ENCODER_N_UNITS[i], activation=ACT_FUNC, 
                        dropout=DROPOUT, recurrent_dropout=DROPOUT, 
                        return_state=True, return_sequences=(i < (len(ENCODER_N_UNITS)-1)), 
                    ) for i in range(len(ENCODER_N_UNITS))
                ],
                "decoder": [
                    functools.partial(BLOCKS_DECODER[DECODER], 
                        n_units=DECODER_N_UNITS[i], activation=ACT_FUNC, 
                        dropout=DROPOUT, recurrent_dropout=DROPOUT, 
                        return_state=True, return_sequences=(i < (len(DECODER_N_UNITS)-1)), 
                    ) for i in range(len(DECODER_N_UNITS))
                ],
            }, 
            "compile_params": {
                "loss_fn": SparseCategoricalCrossentropy(),
                "optimizer": Adam(),
                "metrics": [SparseCategoricalAccuracy()]
            }
        }
    }

    # Get Train Val Dataset
    DATASET, DATASET_ENCODED = LoadTrainDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    inputs["dataset_encoded"] = DATASET_ENCODED

    # Build Model
    X_shape = DATASET_ENCODED["train"]["encoder_input"].shape
    Y_shape = DATASET_ENCODED["train"]["decoder_output"].shape
    MODEL = Model_EncoderDecoderBlocks(
        X_shape=X_shape, Y_shape=Y_shape, 
        Blocks=inputs["model"]["blocks"],
        encoder={
            "embedding_size": ENCODER_EMBEDDING_SIZE
        }, 
        decoder={
            "embedding_size": DECODER_EMBEDDING_SIZE
        },
        use_attention=USE_ATTENTION
    )
    MODEL = Model_Compile(MODEL, **inputs["model"]["compile_params"])

    # Train Model
    TRAINED_MODEL, TRAIN_HISTORY = Model_Train(MODEL, inputs, N_EPOCHS, wandb_data, best_model_path=PATH_BESTMODEL)

    # Load Best Model
    TRAINED_MODEL = Model_LoadModel(PATH_BESTMODEL)
    # Get Test Dataset
    DATASET_TEST, DATASET_ENCODED_TEST = LoadTestDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    # Test Best Model
    loss_test, eval_test, eval_test_inference = Model_Test(
        TRAINED_MODEL, DATASET_ENCODED_TEST,
        target_chars=DATASET_ENCODED_TEST["chars"]["target_chars"],
        target_char_map=DATASET_ENCODED_TEST["chars"]["target_char_map"]
    )

    # Wandb log test data
    wandb.log({
        "loss_test": loss_test,
        "eval_test": eval_test,
        "eval_test_inference": eval_test_inference
    })

    # Close Wandb Run
    # run_name = "ep:"+str(N_EPOCHS) + "_" + "bs:"+str(BATCH_SIZE) + "_" + "nf:"+str(N_FILTERS) + "_" + str(DROPOUT)
    # wandb.run.name = run_name
    wandb.finish()

# Runner Functions
def Runner_ParseArgs():
    '''
    Parse Args
    '''
    global DATASET_PATH_DAKSHINA_TAMIL
    
    parser = argparse.ArgumentParser(description="Training and Testing for DL Assignment 3")

    parser.add_argument("--mode", "-m", type=str, default="train", help="train | test")
    parser.add_argument("--model", "-ml", type=str, default="Models/Model.h5", help="Model path to use or save to")
    parser.add_argument("--dataset", "-dt", type=str, default=DATASET_PATH_DAKSHINA_TAMIL, help="Dataset path to use")

    # Train Args
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")

    parser.add_argument("--encoder", "-en", type=str, default="LSTM", help="Encoder type")
    parser.add_argument("--decoder", "-de", type=str, default="LSTM", help="Decoder type")
    parser.add_argument("--encoder_embedding_size", "-es", type=int, default=64, help="Encoder embedding size")
    parser.add_argument("--decoder_embedding_size", "-des", type=int, default=64, help="Decoder embedding size")
    parser.add_argument("--encoder_n_units", "-eu", type=str, default="64", help="Encoder Num units")
    parser.add_argument("--decoder_n_units", "-du", type=str, default="64", help="Decoder Num units")
    parser.add_argument("--act_func", "-af", type=str, default="relu", help="Activation function")
    parser.add_argument("--dropout", "-d", type=float, default=0.2, help="Dropout")

    args = parser.parse_args()
    DATASET_PATH_DAKSHINA_TAMIL = str(args.dataset).rstrip("/") + "/ta/lexicons/ta.translit.sampled.{}.tsv"
    return args

def Runner_Train(args):
    '''
    Train Model
    '''
    # Load Wandb Data
    WANDB_DATA = json.load(open("config.json", "r"))
    # Sweep Setup
    SWEEP_CONFIG = {
        "name": "run-1",
        "method": "grid",
        "metric": {
            "name": "val_accuracy",
            "goal": "maximize"
        },
        "parameters": {
            "n_epochs": {
                "values": [args.epochs]
            },
            "batch_size": {
                "values": [args.batch_size]
            },

            "encoder": {
                "values": [args.encoder]
            },
            "decoder": {
                "values": [args.decoder]
            },
            "encoder_embedding_size": {
                "values": [args.encoder_embedding_size]
            },
            "decoder_embedding_size": {
                "values": [args.decoder_embedding_size]
            },
            "encoder_n_units": {
                "values": [
                    [int(x) for x in args.encoder_n_units.split(",")]
                ]
            },
            "decoder_n_units": {
                "values": [
                    [int(x) for x in args.decoder_n_units.split(",")]
                ]
            },
            "act_func": {
                "values": [args.act_func]
            },
            "dropout": {
                "values": [args.dropout]
            }
        }
    }
    # Run Sweep
    sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"])
    # sweep_id = ""
    TRAINER_FUNC = functools.partial(Model_Sweep_Run, wandb_data=WANDB_DATA)
    wandb.agent(sweep_id, TRAINER_FUNC, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"], count=1)
    # Save Model
    Model_SaveModel(Model_LoadModel(PATH_BESTMODEL), args.model)

def Runner_Test(args):
    '''
    Test Model
    '''
    # Load Model
    TRAINED_MODEL = Model_LoadModel(args.model)
    # Get Test Dataset
    DATASET_TEST, DATASET_ENCODED_TEST = LoadTestDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    # Test Best Model
    loss_test, eval_test, eval_test_inference = Model_Test(
        TRAINED_MODEL, DATASET_ENCODED_TEST,
        target_chars=DATASET_ENCODED_TEST["chars"]["target_chars"],
        target_char_map=DATASET_ENCODED_TEST["chars"]["target_char_map"]
    )
    # Display
    print("MODEL TEST:")
    print("Loss:", loss_test)
    print("Accuracy:", eval_test)
    print("Accuracy Inference:", eval_test_inference)

# Run
if __name__ == "__main__":
    # Parse Args
    ARGS = Runner_ParseArgs()
    # Run
    if ARGS.mode == "train":
        Runner_Train(ARGS)
    elif ARGS.mode == "test":
        Runner_Test(ARGS)
    else:
        print("Invalid Mode!")