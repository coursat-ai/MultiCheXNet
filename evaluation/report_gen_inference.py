import numpy as np

def tokens_to_text(tokens,tok,end_token='endseq'):
    sentence=""
    for token in tokens:
        if token ==0:
            break
        
        word = tok.index_word[token]
        
        if word==end_token:
            break
            
        sentence+= word+" "
        
    sentence = sentence.strip()
    
    return sentence


def greedy_inference(input_img, tok,encoder_model, decoder_model,max_len,start_token="startseq",end_token='endseq'):
    hidden_layer  =encoder_model(np.expand_dims(input_img,axis=0))
    word = tok.word_index[start_token]
    
    words = []
    
    for index in range(max_len):
        word_probs , hidden_layer = decoder_model.predict([[np.array([word]),hidden_layer]])
        hidden_layer=hidden_layer[0]
        word = np.argmax(word_probs)
        try:
            if tok.index_word[word]==end_token:
                break
        except:
            pass
            
        words.append(word)
        
    words = tokens_to_text(words,tok)
    return words
