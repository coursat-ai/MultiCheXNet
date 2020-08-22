import numpy as np
from nltk.translate.bleu_score import corpus_bleu

def get_predictions_from_data_loader(data_loader,tok,encoder_model, decoder_model,max_len,start_token="startseq",end_token='endseq', inference_type='greedy'):
    
    data_loader_iterator = data_loader.__iter__()
    
    pred_sentences = []
    Gt_sentences = []
    for index, (X,Y) in enumerate(data_loader_iterator):
        for img,_,sample_y in zip(X[0],X[1],Y):
            
            if inference_type=='greedy':
                pred_sentence = greedy_inference(img, tok,encoder_model, decoder_model,max_len)
            
            GT_sentence   = tokens_to_text(sample_y,tok)
            
            pred_sentences.append(pred_sentence)
            Gt_sentences.append(GT_sentence)
        
        if index == data_loader.nb_iteration -1:
            break
        
    return Gt_sentences, pred_sentences

def calculate_bleu_evaluation(GT_sentences, predicted_sentences):
    BLEU_1 = corpus_bleu(GT_sentences, predicted_sentences, weights=(1.0, 0, 0, 0))
    BLEU_2 = corpus_bleu(GT_sentences, predicted_sentences, weights=(0.5, 0.5, 0, 0))
    BLEU_3 = corpus_bleu(GT_sentences, predicted_sentences, weights=(0.3, 0.3, 0.3, 0))
    BLEU_4 = corpus_bleu(GT_sentences, predicted_sentences, weights=(0.25, 0.25, 0.25, 0.25))
    
    return BLEU_1,BLEU_2,BLEU_3,BLEU_4
   
def evaluate_from_dataloader(data_loader,tok,encoder_model, decoder_model,max_len,start_token="startseq",end_token='endseq', inference_type='greedy'):
    Gt_sentences, pred_sentences = get_predictions_from_data_loader(data_loader,tok,encoder_model, decoder_model,max_len,start_token=start_token,end_token=end_token, inference_type=inference_type)
    BLEU_1,BLEU_2,BLEU_3,BLEU_4 = calculate_bleu_evaluation(Gt_sentences, pred_sentences)
    
    return BLEU_1,BLEU_2,BLEU_3,BLEU_4
