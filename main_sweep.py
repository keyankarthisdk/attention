'''
Main Sweep
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
    DECODER = ENCODER
    ENCODER_EMBEDDING_SIZE = config.encoder_embedding_size
    DECODER_EMBEDDING_SIZE = ENCODER_EMBEDDING_SIZE
    ENCODER_N_UNITS = config.encoder_n_units
    DECODER_N_UNITS = ENCODER_N_UNITS
    ACT_FUNC = config.act_func
    DROPOUT = config.dropout
    USE_ATTENTION = wandb_data["attention"]
    LOSS_FN = wandb_data["loss_fn"]

    print("RUN CONFIG:")
    pprint(config)
    print("OTHER CONFIG:")
    pprint(wandb_data)

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
                        return_state=True, return_sequences=True, 
                    ) for i in range(len(DECODER_N_UNITS))
                ],
            }, 
            "compile_params": {
                "loss_fn": LOSS_FUNCTIONS[LOSS_FN](),
                # CategoricalCrossentropy(),
                # SparseCategoricalCrossentropy(),
                "optimizer": Adam(),
                "metrics": ["accuracy"]
            }
        }
    }

    # Get Train Val Dataset
    DATASET, DATASET_ENCODED = LoadTrainDataset_Dakshina(
        DATASET_PATH_DAKSHINA_TAMIL
    )
    inputs["dataset_encoded"] = DATASET_ENCODED
    inputs["dataset_encoded"]["train"]["batch_size"] = BATCH_SIZE
    inputs["dataset_encoded"]["val"]["batch_size"] = BATCH_SIZE

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
    TRAINED_MODEL, TRAIN_HISTORY = Model_Train(
        MODEL, inputs, N_EPOCHS, wandb_data, 
        best_model_path=PATH_BESTMODEL
    )

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

    print("MODEL TEST:")
    print("Loss:", loss_test)
    print("Eval:", eval_test)
    print("Eval Inference:", eval_test_inference)

    # Close Wandb Run
    # run_name = "ep:"+str(N_EPOCHS) + "_" + "bs:"+str(BATCH_SIZE) + "_" + "nf:"+str(N_FILTERS) + "_" + str(DROPOUT)
    # wandb.run.name = run_name
    wandb.finish()

# Run
# if __name__ == "__main__":
#     import traceback
#     DATASET_PATH_DAKSHINA_TAMIL = "Dataset/dakshina_dataset_v1.0/ta/lexicons/ta.translit.sampled.{}.tsv"
#     # Load Wandb Data
#     WANDB_DATA = json.load(open("config.json", "r"))
#     WANDB_DATA.update({
#         "attention": False,
#         "loss_fn": "sparse_categorical_crossentropy" # "categorical_crossentropy", sparse_categorical_crossentropy"
#     })
#     # Sweep Setup
#     SWEEP_CONFIG = {
#         "name": "test-run-1",
#         "method": "grid",
#         "metric": {
#             "name": "val_accuracy",
#             "goal": "maximize"
#         },
#         "parameters": {
#             "n_epochs": {
#                 "values": [10]
#             },
#             "batch_size": {
#                 "values": [128, 256]
#             },

#             "encoder": {
#                 "values": ["LSTM"]
#             },
#             # "decoder": {
#             #     "values": ["LSTM"]
#             # },
#             "encoder_embedding_size": {
#                 "values": [64]
#             },
#             # "decoder_embedding_size": {
#             #     "values": [64]
#             # },
#             "encoder_n_units": {
#                 "values": [
#                     [64],
#                     [64, 64]
#                 ]
#             },
#             # "decoder_n_units": {
#             #     "values": [
#             #         [64],
#             #         [64, 64]
#             #     ]
#             # },
#             "act_func": {
#                 "values": ["sigmoid", "tanh"]
#             },
#             "dropout": {
#                 "values": [0.1, 0.2]
#             }
#         }
#     }

#     try:
#         # Run Sweep
#         sweep_id = wandb.sweep(SWEEP_CONFIG, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"])
#         # sweep_id = ""
#         TRAINER_FUNC = functools.partial(Model_Sweep_Run, wandb_data=WANDB_DATA)
#         wandb.agent(sweep_id, TRAINER_FUNC, project=WANDB_DATA["project_name"], entity=WANDB_DATA["user_name"], count=1)
#         # Save Model
#         # Model_SaveModel(Model_LoadModel(PATH_BESTMODEL), "Models/Model.h5")
#     except Exception as e:
#         # exit gracefully, so wandb logs the problem
#         print(traceback.print_exc())