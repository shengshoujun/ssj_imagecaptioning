import torch
from config import DEVICE as device
from tqdm.auto import tqdm
from utils import LrScheduler, EarlyStopping, SaveBestModel
import numpy as np
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import config
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.spice.spice import Spice
import logging
import datetime
now = datetime.datetime.now()
current_time = now.strftime("%Y-%m-%d-%H-%M-%S")
log_filename = 'training-{}.log'.format(current_time)


nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
def fit(model, train_loader, optimizer, criterion):
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

    loss_list = []
        
    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True, desc="Training Set: ")
    for i, (source, target, attention_mask) in prog_bar:
        # load data and labels to device
        source = source.to(device)
        target = target.to(device)
        attention_mask = attention_mask[:, :-1].to(device)
        target_input = target[:, :-1].clone()
        out = model(source, target_input, attention_mask)
        out = out.reshape(-1, out.shape[2])
        target_out = target[:, 1:].reshape(-1)
        optimizer.zero_grad()
        loss = criterion(out, target_out)
        logging.info(f"Batch : {i+1}/{len(train_loader)}, Loss: {loss.item()}")
        if i%10000==0 and i !=0:
            torch.save({'model_state_dict' : model.state_dict()}, f'bestmodel_{i}.pth')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        loss_list.append(loss.item())
        prog_bar.set_description(f"Batch : {i+1}/{len(train_loader)}")
        prog_bar.set_postfix(loss=loss.item())

    train_loss = np.mean(loss_list) 
    
    return train_loss


def validate(model, val_loader, criterion):
    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []
    meteor_scores = []
    rouge_l_scores = []
    cider_scores = []
    spice_scores = []
    rouge = Rouge()
    cider = Cider()
    meteor = Meteor()
    spice = Spice()
    smooth = SmoothingFunction()
    with torch.no_grad():
        val_losses = []
        prog_bar = tqdm(enumerate(val_loader), total=len(val_loader), leave=True, desc="Validation Set: ")
 
        for i, (source, target, attention_mask) in prog_bar:
            reference = {}
            hypothesis = {}
            source = source.to(device)
            target = target.to(device)
            attention_mask = attention_mask[:, :-1].to(device)
            target_input = target[:, :-1].clone()
            out = model(source, target_input, attention_mask)
            out = out.reshape(-1, out.shape[2])
            target_out = target[:, 1:].reshape(-1)
            loss = criterion(out, target_out)
            val_losses.append(loss.item())
            
            # Convert target and output tensor to list of sentences
            target_sentences = config.TOKENIZER.batch_decode(target_out.tolist(), skip_special_tokens=True)
            out_sentences = config.TOKENIZER.batch_decode(out.argmax(dim=-1).tolist(), skip_special_tokens=True)
  
            # Calculate BLEU scores
            for ref, hyp in zip(target_sentences, out_sentences):
                ref = nltk.word_tokenize(ref.lower())
                hyp = nltk.word_tokenize(hyp.lower())
                bleu1_scores.append(sentence_bleu([ref], hyp, weights=(1, 0, 0, 0),smoothing_function=smooth.method1))
                bleu2_scores.append(sentence_bleu([ref], hyp, weights=(0.5, 0.5, 0, 0),smoothing_function=smooth.method2))
                bleu3_scores.append(sentence_bleu([ref], hyp, weights=(0.33, 0.33, 0.33, 0),smoothing_function=smooth.method3))
                bleu4_scores.append(sentence_bleu([ref], hyp, weights=(0.25, 0.25, 0.25, 0.25,),smoothing_function=smooth.method4))
                meteor_scores.append(meteor_score([ref], hyp))
            for id, (ref, hyp) in enumerate(zip(target_sentences, out_sentences)):
                reference[id] = [ref]
                hypothesis[id] = [hyp]
        
            # spice_scores.append(spice.compute_score(reference,hypothesis))

            prog_bar.set_description(f"Batch : {i+1}/{len(val_loader)}")
            prog_bar.set_postfix(loss=loss.item())
        rouge_l_scores, _ = rouge.compute_score(reference, hypothesis)
        cider_scores, _ = cider.compute_score(reference, hypothesis)
        val_loss = np.mean(val_losses)
        bleu1_score = np.mean(bleu1_scores)
        bleu2_score = np.mean(bleu2_scores)
        bleu3_score = np.mean(bleu3_scores)
        bleu4_score = np.mean(bleu4_scores)
     
        meteor_sc = np.mean(meteor_scores)
        rouge_l_score = np.mean(rouge_l_scores)
        cider_sc = np.mean(cider_scores)
        spice_sc = np.mean(0)
    return val_loss, bleu1_score, bleu2_score, bleu3_score, bleu4_score,meteor_sc,rouge_l_score,cider_sc,spice_sc



def train(model, train_loader, val_loader, num_epochs, learning_rate, criterion, optimizer, early_stop=False):
    train_loss_list = []
    save_best_model = SaveBestModel("bestmodel")
    lr_scheduler = LrScheduler(optimizer, patience=1)
    if early_stop:
        early_stopping = EarlyStopping(patience=10)

    for epoch in range(num_epochs):

        train_loss = fit(model, train_loader, optimizer, criterion)
        save_best_model(train_loss, epoch, model, optimizer, criterion)
        val_loss, bleu1_score, bleu2_score, bleu3_score, bleu4_score,meteor,rouge_l,cider,spice = validate(model, val_loader, criterion)
        train_loss_list.append(train_loss)
  

        print(f"""Training Set :\nEpoch :{epoch+1}/{num_epochs}, \tloss : {train_loss:.3f},\tLearning Rate : {learning_rate}""")
        print(f"""Validation Set :\nEpoch :{epoch+1}/{num_epochs}, \tloss : {val_loss:.3f}""")
        print(f"Validation Set BLEU Scores:\nBLEU-1: {bleu1_score:.3f}, BLEU-2: {bleu2_score:.3f}, BLEU-3: {bleu3_score:.3f}, BLEU-4: {bleu4_score:.3f}, meteor: {meteor:.3f}, rouge_l: {rouge_l:.3f}, cider: {cider:.3f}, spice: {spice:.3f}")

        logging.info(f"Training Set :\nEpoch :{epoch+1}/{num_epochs}, \tloss : {train_loss:.3f},\tLearning Rate : {learning_rate}")
        logging.info(f"""Validation Set :\nEpoch :{epoch+1}/{num_epochs}, \tloss : {val_loss:.3f}""")
        logging.info(f"Validation Set BLEU Scores:\nBLEU-1: {bleu1_score:.3f}, BLEU-2: {bleu2_score:.3f}, BLEU-3: {bleu3_score:.3f}, BLEU-4: {bleu4_score:.3f}, meteor: {meteor:.3f}, rouge_l: {rouge_l:.3f}, cider: {cider:.3f}, spice: {spice:.3f}")
        lr_scheduler(train_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print("--"*40)
        if early_stopping:
            early_stopping(train_loss)
            if early_stopping.early_stop:
                break

    return model, train_loss_list, val_loss