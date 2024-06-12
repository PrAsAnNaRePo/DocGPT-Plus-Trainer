from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import deepspeed
from tqdm import tqdm
from doc_dataset import DocDataset

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def count_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + nontrainable_params

    if total_params / 1e9 > 1:
        total_params_str = f'{total_params / 1e9:.1f}B'
    else:
        total_params_str = f'{total_params / 1e6:.1f}M'

    logger.info(f'Number of trainable parameters: {trainable_params}')
    logger.info(f'Number of non-trainable parameters: {nontrainable_params}')
    logger.info(f'Total number of parameters: {total_params_str}')


def add_argument():
    parser=argparse.ArgumentParser(description='DocGPT')

    parser.add_argument("--model_id", type=str, default="microsoft/Phi-3-mini-128k-instruct")
    parser.add_argument("--epochs", type=int, default=2, help="No. of epochs")
    parser.add_argument("--max_len", type=int, default=512, help="maximum context length of the model (to train with).")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--ds_config", default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    
    parser = deepspeed.add_config_arguments(parser)
    args=parser.parse_args()

    return args


def initialize(args,
               model,
               optimizer=None,
               parameters=None,
               training_data=None,
               ):
    parameters = filter(lambda p: p.requires_grad, model.parameters()) if parameters is None else parameters
    
    model_engine, optimizer, trainloader, _ = deepspeed.initialize(args=args, model=model, model_parameters=parameters, training_data=training_data)
    return model_engine, optimizer, trainloader


def train(args, model_engine, train_loader):
    for epoch in range(args.epochs):
        loop = tqdm(train_loader, leave=True)
        for batch_idx, batch in enumerate(loop):
        
            input_ids = batch["input_ids"].reshape(-1, args.max_len).to(model_engine.local_rank)
            attention_mask = batch["attention_mask"].reshape(-1, args.max_len).to(model_engine.local_rank)

            output = model_engine(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = output.loss
            model_engine.backward(loss)
            model_engine.step()

            if batch_idx % args.log_interval == 0:
                logger.info(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        model_engine.save_checkpoint(f"checkpoints")

def main():
    args = add_argument()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    count_parameters(model)
    
    training_data = DocDataset(args, tokenizer=tokenizer)

    model_engine, optimizer, train_loader = initialize(
        args=args,
        model=model,
        training_data=training_data,
    )
    train(args, model_engine, train_loader)

if __name__ == "__main__":
    main()
