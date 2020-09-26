## -----------------------------------------------------------------------------
### Goal: Setup a model to perform image captioning on a given image.
##        Accepts parameters via the CL.
##        NOTES:
##              1) The encoder is a Keras Google Inception-v3 model pre-trained on Imagenet.
##              2) Decoder is setup as coded below and the weights are populated from a locally saved file.
##              3) The Word2Index and Index2Word data structure are reloaded from pickle file locations.
##                 These are hard-coded - CHANGE IF REQUIRED BEFORE RUNNING.
##              4) If in debug mode, each image will be briefly displayed before proceeding to generate 
##                 inference using the decoder.
##              5) Expects the results from two previous stages to be sent. This will be combined for processing by
##                 this stage. The two previous stages are:
##                 a) Id Key Elements - final selected words by user for each sentence that were used to query Neo4j
##                 b) GUI Candidate Image Selection - final images selected by user from the images output by Neo4j query
##
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -inimg    :image to be inferred
## 2) -inwtf   :weights file for the decoder
## -----------------------------------------------------------------------------
## Usage example:
##    python3 ImgCap_Reload_Lappy_Infer_2.py -img "/Path/Image/ToInfer/filename.jpg" -wtfile "/Path/PickleFile/WeightsDecoder/filename.h5"
## -----------------------------------------------------------------------------

from __future__ import print_function

import os
#import sys
import copy
import json
#import time
#import datetime
#import string
import numpy as np
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import layers
#from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
#import re
import pickle
#import itertools
#import PIL
#import PIL.Image

import argparse

argparser = argparse.ArgumentParser(
    description='perform image captioning on given image')

argparser.add_argument(
    '-img',
    '--image_input',
    type=str,
    help='absolute path of input image file with .jpg extension')

argparser.add_argument(
    '-wtfile',
    '--weights_file_input',
    type=str,
    help='absolute path of the weights file to setup decoder')


def preprocess_image_for_Incepv3(_img_path, _key = 'DUMMY', _DEBUG_SWITCH=False):
    """
    ## Make images suitable for use by Inception-v3 model later
    ##
    ## Resize to (299, 299)
    ## As model needs 4-dim input tensor, add one dimenion to make it (1, 299, 299, 3)
    ## Preprocess the image using custom function of Inception-v3 model
    """
    img = tf.keras.preprocessing.image.load_img(_img_path, target_size=(299, 299))
    #print(f"type={type(img)}") # type(img): type=<class 'PIL.Image.Image'>
    if _DEBUG_SWITCH:
        plt.figure(figsize=(12,6))
        plt.subplot(121)
        plt.title('Original Image(Resized): ' + _key + '.jpg')
        plt.imshow(img)
    img = tf.keras.preprocessing.image.img_to_array(img) # Converts PIL Image instance to numpy array (299,299,3)
    img = np.expand_dims(img, axis=0) #Add one more dimension: (1, 299, 299, 3) # Inception-V3 requires 4 dimensions
    img = tf.keras.applications.inception_v3.preprocess_input(img) # preprocess image as per Inception-V3 model
    if _DEBUG_SWITCH:
        plt.subplot(122)
        plt.title('Preprocessed image for Inception-V3: ' + _key + '.jpg')
        plt.imshow(img[0])
    return img  # shape will be (1, 299, 299, 3)

def encode_image(_imgpath, _model_CNN_encoder, _key = 'DUMMY', _DEBUG_SWITCH = False):
    """
    # Function to encode given image into a vector of size (2048, )
    """
    preproc_img = preprocess_image_for_Incepv3(_imgpath, _key = 'DUMMY', _DEBUG_SWITCH = False) # preprocess image per Inception-v3 requirements
    encoded_features = _model_CNN_encoder.predict(preproc_img) # Get encoding vector for image
    encoded_features = encoded_features.reshape(encoded_features.shape[1], ) # reshape from (1, 2048) to (2048, )
    return encoded_features

def reload_rnn_encoder_saved_weights(_saved_weights_file, _EMBEDDING_DIMS, _VOCAB_SIZE, _MAX_LENGTH_CAPTION, _DEBUG_SWITCH = False):
    if os.path.exists(_saved_weights_file) and os.path.isfile(_saved_weights_file):
        ## Decoder Model defining
        
        ## parameters to define model
        #EMBEDDING_DIMS is initialised earlier while creating embedding matrix
        #VOCAB_SIZE is initialised earlier
        #MAX_LENGTH_CAPTION is initialised earlier
        
        inputs1 = keras.Input(shape=(2048,))
        fe1 = keras.layers.Dropout(0.5)(inputs1)
        fe2 = keras.layers.Dense(256, activation='relu')(fe1)
        
        # partial caption sequence model
        inputs2 = keras.Input(shape=(_MAX_LENGTH_CAPTION,))
        se1 = keras.layers.Embedding(_VOCAB_SIZE, _EMBEDDING_DIMS, mask_zero=True)(inputs2)
        se2 = keras.layers.Dropout(0.5)(se1)
        se3 = keras.layers.LSTM(256)(se2)
        
        # decoder (feed forward) model
        decoder1 = keras.layers.add([fe2, se3])
        decoder2 = keras.layers.Dense(256, activation='relu')(decoder1)
        outputs = keras.layers.Dense(_VOCAB_SIZE, activation='softmax')(decoder2)
        
        # merge the two input models
        reloaded_rnn_decoder_model = keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
        
        if _DEBUG_SWITCH:
            print(f"\nRNN Decoder model defined with these paramenters:\nEMBEDDING_DIMS = {_EMBEDDING_DIMS} , VOCAB_SIZE = {_VOCAB_SIZE} , MAX_LENGTH_CAPTION = {_MAX_LENGTH_CAPTION}\nAttempting to load weights...")
        
        ## load the weights
        reloaded_rnn_decoder_model.load_weights(_saved_weights_file)

        if _DEBUG_SWITCH:
            print(f"SUCCESS - Reloaded weights from :: {_saved_weights_file}")
        return reloaded_rnn_decoder_model
    else:
        print(f"\nERROR reloading weights. Check weights file exists here = {_saved_weights_file} ;\nOR model setup parameters incompatible with the saved weights file given.")
        return None

def greedySearch(_decoder_model, _img_encoding, _max_length, _wordtoix = None, _ixtoword = None):
    wordtoix = _wordtoix
    ixtoword = _ixtoword
    in_text = 'startseq'
    for i in range(_max_length):
        sequence = [ wordtoix[w] for w in in_text.split() if w in wordtoix ]
        sequence = keras.preprocessing.sequence.pad_sequences([sequence], maxlen=_max_length)
        yhat = _decoder_model.predict([_img_encoding,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    caption_out = in_text.split()
    #caption_out = caption_out[1:-1]  ## drop the startseq and endseq words at either end
    caption_out = ' '.join(caption_out)
    return caption_out

"""
def do_inference_one_image(_infer_idx_pos, _imgs_arr, _IPDIRIMGS, _MAX_LENGTH_CAPTION, _model_RNN_decoder, _descriptions):
    infer_image_path = _IPDIRIMGS + _imgs_arr[_infer_idx_pos] + '.jpg'
    
    ## show the original image
    image = img_encodings[ _imgs_arr[_infer_idx_pos] ].reshape((1,2048))
    x = plt.imread(infer_image_path)
    plt.imshow(x)
    plt.show()
    
    ## get the prediction caption using greedy search
    predicted_caption = greedySearch(_model_RNN_decoder, image, _MAX_LENGTH_CAPTION)
    print(f"\nFor image :: {infer_image_path}\n\nInference caption output:\n{ predicted_caption }")
    print("")
    ## show the original captions
    for idx, orig_cap in enumerate(_descriptions.get(_imgs_arr[_infer_idx_pos])):
        print(f"Original caption {idx+1}  :::  {orig_cap}")
"""

def do_inference_one_image_totally_new(_infer_image_jpg, _img_encoding, _MAX_LENGTH_CAPTION, _model_RNN_decoder, _DEBUG_SWITCH = False, _wordtoix = None, _ixtoword = None ):
    ## show the original image
    image = _img_encoding.reshape((1,2048))
    
    ## briefly show the image about to be inferred if in debug mode
    if _DEBUG_SWITCH:
        plt.ion()
        x = plt.imread(_infer_image_jpg)
        plt.imshow(x)
        plt.show()
        plt.pause(1.5)  ## display for 1.5 seconds
    
    predicted_caption = None
    ## get the prediction caption using greedy search
    predicted_caption = greedySearch(_model_RNN_decoder, image, _MAX_LENGTH_CAPTION, _wordtoix = _wordtoix, _ixtoword = _ixtoword)
    
    if _DEBUG_SWITCH:
        print(f"\nFor image :: {_infer_image_jpg}\n\nInference caption output:\n{ predicted_caption }")
    return predicted_caption

def infer_for_caption_one_image(_in_img_jpg, _MAX_LENGTH_CAPTION, _model_CNN_encoder, _model_RNN_decoder, _wordtoix = None, _ixtoword = None, _DEBUG_SWITCH = False):
    ## Get the encoding by running thru bottleneck
    ## Function to encode given image into a vector of size (2048, )
    img_encoding_for_inference = encode_image(_in_img_jpg, _model_CNN_encoder, _DEBUG_SWITCH)
    
    if _DEBUG_SWITCH:
        print(f"Encoding shape = {img_encoding_for_inference.shape}")
    
    ## now do the decoder inference using the encoding
    predicted_caption = do_inference_one_image_totally_new(_in_img_jpg, img_encoding_for_inference, _MAX_LENGTH_CAPTION, _model_RNN_decoder, _DEBUG_SWITCH, _wordtoix = _wordtoix, _ixtoword = _ixtoword )
    return predicted_caption

def convert_prev_stages_data(_id_elements_module_results, _gui_candidate_image_selection_module_results):
    """
    ## Explanation of the data preparation using previous stage outputs:
    ## 
    ## The main variable is a list of exactly 3 items - one for each of the original input sentence.
    ##     Each of these entries in the list is a dict with two keys: 'key_elements' and 'selected_images'

    ## 'key_elements' is a list of objects selected by user at end of the Id Key Elements stage
    ##      - 0 to 3 values - can be empty list too
    ## 'selected_images' is a list containing lists. The inner list represents each of the images finally 
    ##      selected by the user at end GUI Selection of Images that the Neo4j query had returned.
    ##      - 0 to 5 images - can be empty list too
    ##      Regarding innner list: first entry is the image path, second is initialised as None to be filled 
    ##          later with the caption after processing the image through this stage.
    ##          Thus, ['/full/path/to/image.jpg' , None] will become something like
    ##                ['/full/path/to/image.jpg' , 'caption of the images after processing']

    ## Example of how the data structure could be in this scenario:
    ## For input sentence 1, the user finally selected 2 Key Elements, then out of the up to 20 images
    ##     returned by the neo4j query, user selected only 3 images to send to captioning stage while
    ##     up to 5 could have been selected.
    ## For input sentence 2, the user finally selected 0 Key Elements, thus there were no images
    ##     returned by the neo4j query, and nothing for the user to select and to send to captioning stage.
    ## For input sentence 3, the user finally selected 3 Key Elements, then out of the up to 20 images
    ##     returned by the neo4j query, user selected 0 images to send to captioning stage while
    ##     up to 5 could have been selected.
    ## -----------------------------------------------------------------------------
    example_data_usable_by_img_captioning_functionality = \
    [
        {
            'key_elements' : ['q1_obj1', 'q1_obj2'],
            'selected_images' : [
                ['/path/to/q1_image1.jpg' , None],
                ['/path/to/q1_image2.jpg' , None],
                ['/path/to/q1_image3.jpg' , None]
            ]
        },
        {
            'key_elements' : [],
            'selected_images' : []
        },
        {
            'key_elements' : ['q3_obj1', 'q3_obj2', 'q3_obj3'],
            'selected_images' : []
        }
    ]
    ## -----------------------------------------------------------------------------
    """
    ## dictionary with key as the image source, value is the location of that datasets images
    source_and_location_image_datasets = {
        'flickr30k' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/flickr30k_images/flickr30k_images/',
        'coco_val_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_val2017_5k/val2017/',
        'coco_test_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/',
        'coco_train_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_train2017_118k/'
        }
    
    ## add the full path to the image files in a NEW data structure copy
    new_gui_candidate_image_selection_module_results = copy.deepcopy(_gui_candidate_image_selection_module_results)
    for each_query_images in new_gui_candidate_image_selection_module_results:
        for each_img_dict in each_query_images:
            dataset_path = source_and_location_image_datasets[each_img_dict['Source']]
            each_img_dict['Image'] = dataset_path + each_img_dict['Image']
    
    ## create the final data structure and return it
    data_for_img_cap_functionality = list()
    for query_key_elements_info, query_images_selected_info in zip(_id_elements_module_results, new_gui_candidate_image_selection_module_results):
        new_entry = dict()
        new_entry['key_elements'] = copy.deepcopy(query_key_elements_info)
        new_entry['selected_images'] = list()
        for each_img_info_dict in query_images_selected_info:
            new_entry['selected_images'].append( [each_img_info_dict['Image'] , None] )
        data_for_img_cap_functionality.append(new_entry)
    return data_for_img_cap_functionality

def _main_(args):
    DEBUG_SWITCH = True

    ### TEMP CODE FOR TESTING - START - these variables will be passed directly to functionality in the combined logic script

    ## data from ID Key Elements selection stage
    _id_elements_module_results = [[], ['car'], ['person', 'truck']]

    ## data from GUI Selection of Images returned by Neo4j images
    _gui_candidate_image_selection_module_results = [[], [{'Image': '000000033825.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000155796.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000224207.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000313777.jpg', 'Source': 'coco_test_2017'}], [{'Image': '000000169542.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000449668.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000518174.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000361201.jpg', 'Source': 'coco_test_2017'}, {'Image': '000000304424.jpg', 'Source': 'coco_test_2017'}]]
    
    ### TEMP CODE FOR TESTING - END - these variables will be passed directly to functionality in the combined logic script
    
    # process command line arguments
    IMG_TO_INFER        = args.image_input            # -inimg parameter
    SAVED_WEIGHTS_PATH  = args.weights_file_input     # -inwtf parameter

    I2W_FILE = r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCap/SavedData/run2_1_kagg/ixtoword_train_97000.pkl'
    W2I_FILE = r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Code/ModelsRuns/ImgCap/SavedData/run2_1_kagg/wordtoix_train_97000.pkl'

    ## basic sanity checks
    if (not os.path.exists(I2W_FILE) ) or (not os.path.isfile(I2W_FILE )):
        print(f"\nFATAL ERROR: Index to Word dict not found at :: {I2W_FILE}\nPremature Exit with exit code 110")
        exit(110)
    if (not os.path.exists(W2I_FILE) ) or (not os.path.isfile(W2I_FILE )):
        print(f"\nFATAL ERROR: Word to Index dict not found at :: {W2I_FILE}\nPremature Exit with exit code 111")
        exit(111)
    if (not os.path.exists(SAVED_WEIGHTS_PATH) ) or (not os.path.isfile(SAVED_WEIGHTS_PATH )):
        print(f"\nFATAL ERROR: Weights file for Decoder model not found :: {W2I_FILE}\nPremature Exit with exit code 112")
        exit(112)
    if (not os.path.exists(IMG_TO_INFER) ) or (not os.path.isfile(IMG_TO_INFER )):
        print(f"\nFATAL ERROR: Image to be inferred not found :: {IMG_TO_INFER}\nPremature Exit with exit code 113")
        exit(113)
    
    try:
        ### Load pre-trained model of Inception-v3 pretrained on Imagenet
        model_inception_v3_pretrained_imagement = tf.keras.applications.InceptionV3(weights='imagenet')
        # Create new model, by removing last layer (output layer) from Inception-V3
        model_CNN_encoder = keras.Model(inputs=model_inception_v3_pretrained_imagement.input, outputs=model_inception_v3_pretrained_imagement.layers[-2].output)
        ## only for DEBUG
        #type(model_CNN_encoder) ## should be tensorflow.python.keras.engine.functional.Functional
    except Exception as error_encoder_load_msg:
        print(f"\nFATAL ERROR: Could not load pretrained CNN-Encoder Inception-v3 model")
        print(f"Error message :: {error_encoder_load_msg}\nPremature Exit with exit code 120")
        exit(120)
    
    try:
        ## load the ixtoword and wordtoix dicts
        with open(I2W_FILE, 'rb') as handle:
            ixtoword = pickle.load(handle)
        with open(W2I_FILE, 'rb') as handle:
            wordtoix = pickle.load(handle)
        ## only for DEBUG
        if DEBUG_SWITCH:
            print(f"Check wordtoix entries ::\nstartseq = {wordtoix.get('startseq')}\tendseq = {wordtoix.get('endseq')}\tbird = {wordtoix.get('bird')}")
            print(f"Check ixtoword entries ::\nix 1 = {ixtoword.get('1')}\tix 10 = {ixtoword.get('10')}\tix 1362 = {ixtoword.get('1362')}")
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Could not load the Word2Ix and/ or Ix2Word dicts from specified locations ::")
        print(f"Word2Ix :: {W2I_FILE}\nIx2Word :: {I2W_FILE}")
        print(f"Premature Exit with exit code 125")
        exit(125)
    except Exception as error_word_index_dict_load_msg:
        print(f"\nFATAL ERROR: unknown error loading the Word2Ix and/ or Ix2Word dicts :: {error_word_index_dict_load_msg}")
        print(f"Premature Exit with exit code 127")
        exit(127)
    
    ## lengths of the the two dicts MUST be equal
    if len(wordtoix) != len(ixtoword):
        print(f"\nFATAL ERROR: Lengths of Word2Ix and Ix2Word dicts were not equal.")
        print(f"Word2Ix length = {len(wordtoix)}\t\t Ix2Word length = {len(ixtoword)}")
        print(f"Premature Exit with exit code 130")
        exit(130)
    
    try:
        ### Load the Decoder network using the saved weights file
        ## Parameters to use while defining the Decoder model again
        EMBEDDING_DIMS = 200
        VOCAB_SIZE = 6758
        MAX_LENGTH_CAPTION = 49
        reloaded_RNN_decoder = reload_rnn_encoder_saved_weights(SAVED_WEIGHTS_PATH, EMBEDDING_DIMS, VOCAB_SIZE, MAX_LENGTH_CAPTION, DEBUG_SWITCH)
        if DEBUG_SWITCH:
            type(reloaded_RNN_decoder)   ## should say     tensorflow.python.keras.engine.training.Model
    except Exception as error_decoder_load_msg:
        print(f"\nFATAL ERROR: Could not load LSTM-Decoder model.")
        print(f"Error message :: {error_decoder_load_msg}\nPremature Exit with exit code 120")
        exit(120)
    
    ## prepare the data for this functionality using data from earlier two stages
    try:
        data_for_img_cap_functionality = convert_prev_stages_data(_id_elements_module_results, _gui_candidate_image_selection_module_results)
        if DEBUG_SWITCH:
            print(f"\n\ndata_for_img_cap_functionality =\n{data_for_img_cap_functionality}\n\n")
    except Exception as data_prep_this_functionality_msg:
        print(f"\nFATAL ERROR: Problem preparing the data for this functionality using the data passed by previous stage.")
        print(f"\nData sent by previous stages:\n _id_elements_module_results =\n{_id_elements_module_results}\n _gui_candidate_image_selection_module_results =\n{_gui_candidate_image_selection_module_results}\n")
        print(f"Error message :: {data_prep_this_functionality_msg}\nPremature Exit with exit code 130")
        exit(130)
    
    if DEBUG_SWITCH:
        print(f"\n\nBEFORE:\n{data_for_img_cap_functionality}\n\n")
    ## perform the inference and update the caption in the data structure
    for sentence_info in data_for_img_cap_functionality:
        for each_img_info in sentence_info['selected_images']:
            IMG_TO_INFER = each_img_info[0]
            out_caption = infer_for_caption_one_image(IMG_TO_INFER, MAX_LENGTH_CAPTION, model_CNN_encoder, reloaded_RNN_decoder, wordtoix, ixtoword, DEBUG_SWITCH)
            ## strip off the startseq and endseq - note that startseq is ALWAYS present as the first word, but the endseq MAY NOT ALWAYS be at the end
            out_caption = out_caption.split(' ')
            if out_caption[-1] == 'endseq':
                out_caption = ' '.join(out_caption[1:-1]) ## remove both ends
            else:
                out_caption = ' '.join(out_caption[1:])   ## remove only startseq
            if DEBUG_SWITCH:
                print(f"Inference caption = {out_caption}")
            each_img_info[1] = out_caption
    if DEBUG_SWITCH:
        print(f"\n\nAFTER:\n{data_for_img_cap_functionality}\n\n")
    #out_caption = infer_for_caption_one_image(IMG_TO_INFER, MAX_LENGTH_CAPTION, model_CNN_encoder, reloaded_RNN_decoder, wordtoix, ixtoword, DEBUG_SWITCH)
    #print(f"\nFor image :: {IMG_TO_INFER}\n\nInference caption output:\n{out_caption}\n")
    
    return


if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
    print(f"\nDone\n")