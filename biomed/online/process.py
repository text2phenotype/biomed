#this module is used to transform the annotations or corrections we take
#from BRAT tool and transform them to trainable data - sequences
#these sequences can be created by sampling sequences centered around the annotated word
#given a certain window span, we can also weight such sequences differently by
#some measure that can combine the uncertainty score, confidence score or number of corrections on the same annotaitons
#such sequences will be fed to training pool or fed to online training
#each ann data should have a score, each prediction should have a score

#at correction step -> evaluate feedback from piror knowledge of the data -> p(x|theta) -> feedback here
#-> evaluate the value of each annotation files information gain -> look at the annotated places and provide

#high gain, if uncertain document, -> class i, class j different, cross-entropy loss, loss -> gain
#ran cross-entropy loss on this document of some certain model -> de-bias ->

#training step -> online: loss at a certain region. ->

#goal more efficiently training our model and utilize our existing knowledge with new annotations

#entropy -> the confusion/uncertainty that the model is at the decision -> cross entropy, how off or well the model fits
#to the current annotation file ->

#to measure of these document -> weight whole document, high uncertainty, high averaged cross
#entropy according to new annotation
#-> whole

#weight the training data differently combining our existing knowledge -> more confused decision, most wrongs (higher cross
#entropy). -> gated, target where in the weight matrix we should reinforce or re-adjust the weight->

#or just weight the places where there is annotation or correction, not the whole document -> create training pieces.
#and add that as a seperate json file. -> even -> assign a score to each token's annotation weight?

#each token has a score too.

#weight the document according to the loss ->

import numpy

def construct_sample_weight_with_doc_score(y_reshaped_np, doc_score, tokens):
    """
    this function takes in sample weight and prior document score for and token confidence to construct sample weight
    to strengthen our own model
    :param y_reshaped_np: the training label Y
    :param doc_score: a list of document scores, each element is a tuple and first element being a uncertainty score
    :param tokens: the token that has the confidence score in it from the prediction of our own model
    and the second document being the length of the each document
    :return:
    """
    #binary search to look up the weight, number of docs? number of tokens?
    token_entropy_by_doc = [0]*len(tokens)
    curr_position = 0
    for i in range(len(doc_score)):
        last_position = curr_position
        curr_position += doc_score[i][0]
        token_entropy_by_doc[curr_position:last_position] = doc_score[i][1]

    temp_sample_weight = numpy.zeros(y_reshaped_np.shape[0], y_reshaped_np.shape[1])
    for i in range(y_reshaped_np.shape[0]):
        for j in range(y_reshaped_np.shape[1]):
            #reverse to get the token index in the whole sentence list
            #and look up the score that is the this document uncertainty
            token_index = i + j
            #look up the document score from the doc_tokens
            confidence_score = tokens[token_index]['prob']
            doc_entropy = token_entropy_by_doc[token_index]
            temp_sample_weight[i][j] = doc_entropy*(1/confidence_score) #can also be the entropy of that perticular token
            #in the correction scenario which conversion from which class to which class also matters

    return temp_sample_weight

def construct_sample_weight_with_doc_score_class_weight(y_reshaped_np, class_weight, doc_score, tokens):
    """
    this function takes in class weight, and document score to up-weight the specific sequence.
    :param y_reshaped_np:
    :param doc_score:
    :param tokens:
    :return:
    """
    token_entropy_by_doc = [0]*len(tokens)
    curr_position = 0
    for i in range(len(doc_score)):
        last_position = curr_position
        curr_position += doc_score[i][0]
        token_entropy_by_doc[curr_position:last_position] = doc_score[i][1]

    temp_sample_weight = numpy.zeros(y_reshaped_np.shape[0], y_reshaped_np.shape[1])
    for i in range(y_reshaped_np.shape[0]):
        for j in range(y_reshaped_np.shape[1]):
            token_index = i + j
            confidence_score = tokens[token_index]
            doc_entropy = token_entropy_by_doc[token_index]
            temp_sample_weight[i][j] = class_weight[numpy.argmax(y_reshaped_np[i][j])]*doc_entropy*(1/confidence_score)
    return temp_sample_weight
