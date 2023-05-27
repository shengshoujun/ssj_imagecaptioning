import transformers
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import logging
logging.set_verbosity_warning()


DEVICE = "cpu"
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8 # Convert to 16
VALID_BATCH_SIZE = 32
DECODER_DROPOUT = 0.2
EPOCHS = 10
LEARNING_RATE = 3e-5
NUM_LAYERS = 12
NUM_HEADS = 12
DECODER_PATH = "gpt2"
PRE_TRAINED_MODELS_PATH = "./pre_trained_models/"
TOKENIZER = GPT2Tokenizer.from_pretrained(PRE_TRAINED_MODELS_PATH+DECODER_PATH,
                                       bos_token = '<|endoftext|>',
                                       eos_token = '<|endoftext|>',
                                       pad_token = '<|pad|>',
                                       do_lower_case=True)


PRETRAINED_GPT2MODEL = GPT2LMHeadModel.from_pretrained(PRE_TRAINED_MODELS_PATH+DECODER_PATH, 
                                    # output_hidden_states=True, 
                                    embd_pdrop = 0.2, 
                                    attn_pdrop = 0.2,
                                    add_cross_attention=True, 
                                    activation_function='gelu_new').to(DEVICE)
