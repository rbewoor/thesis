### Goal: Process the input file which has the speech to text transcriptions (one per line).
###       Using Spacy, remove stop-words, perform part of speech tagging.
##        Output two files:
###            One file all the words with the comprehensive data.
###            Second has only the noun words. Stored in data structure that the neo4j
###                   query module can process later.
## 
## Command line arguments:
## 1) -ipfile           : location for input file with the stt transcriptions
## 2) -opfileallposinfo : output file with parts-of-speech tagging (POS) for ALL non-stop words in sentences
## 3) -opfilekeyelem   : output file with only nouns in the required data structure that can be used
##                               to query the neo4j database later
## 
## Usage example:
##    python3 id_elements_1.py -ipfile "/location/of/file.txt" -opfileallposinfo "/location/of/file/all_non_stop_Words_pos_info.txt" -opfilekeyelem "/location/of/file/only_nouns.txt"

import argparse
import os
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import json

argparser = argparse.ArgumentParser(
    description='parse the transcriptions, identify key elements and write output file')

argparser.add_argument(
    '-ipfile',
    '--inputfile',
    help='location for input file with the stt transcriptions')

argparser.add_argument(
    '-opfileallposinfo',
    '--opfileallwordsposinfo',
    help='location for output file where the key elements will be stored')

argparser.add_argument(
    '-opfilekeyelem',
    '--opfilekeyelementsonly',
    help='location for output file containing key elements')

def _main_(args):
    # setup spacy
    nlp = spacy.load('en_core_web_lg')

    # process command line arguments
    ipfile            = args.inputfile              # -ipfile parameter
    opfileallposinfo  = args.opfileallwordsposinfo  # -opfileallposinfo parameter
    opfilekeyelem     = args.opfilekeyelementsonly  # -opfilekeyelem parameter
    
    # check valid input file
    if not os.path.isfile(ipfile):
        print(f"FATAL ERROR: Input for ipfile parameter is not an existing file.\nExiting with RC=100")
        exit(100)
    else:
        # read the file, storing each line as an element
        try:
            sentences_orig = [] # sentences to process
            with open(ipfile, 'r') as infile:
                for line in infile:
                    # remove the new line and the period if present
                    sentences_orig.append(line.rstrip("\n").rstrip("."))
            ### DEBUG DEBUG DEBUG - start
            #print(f"\n{sentences_orig}\n")
            ### DEBUG DEBUG DEBUG - end
        except Exception as ipfile_read_error:
            print(f"\n\nFATAL ERROR: Problem reading the input file.\nError message: {ipfile_read_error}\nExiting with RC=200")
            exit(200)
        
    # array for tokens of the original sentences
    sentences_words_list = []
    for _ in range(len(sentences_orig)):
        sentences_words_list.append([])
    
    # array for tokens of original sentence after removing stop words
    sentences_words_list_no_stop = []
    for _ in range(len(sentences_orig)):
        sentences_words_list_no_stop.append([])
    
    # nlp document arrays
    docs = []
    for each_sentence in sentences_orig:
        docs.append(nlp(each_sentence))
    
    # document arrays with stop words removed
    sentences_no_stop = []
    
    ### DEBUG DEBUG DEBUG - start
    print(f"\n\nThe following sentences were read from the input file:\n")
    for idx, each_input_sentence in enumerate(sentences_orig):
        print(f"Sentence {idx+1} :: {each_input_sentence}")
    ### DEBUG DEBUG DEBUG - end
    
    # Create list of word tokens for each sentence doc
    for idx, doc in enumerate(docs):
        for token in doc:
            sentences_words_list[idx].append(token.text)
    
    ## Spacy gives following info for each word as part of POS prcoessing
    #Text: The original word text.
    #Lemma: The base form of the word.
    #POS: The simple UPOS part-of-speech tag.
    #Tag: The detailed part-of-speech tag.
    #Dep: Syntactic dependency, i.e. the relation between tokens.
    #Shape: The word shape – capitalization, punctuation, digits.
    #is alpha: Is the token an alpha character?
    #is stop: Is the token part of a stop list, i.e. the most common words of the language?
    # 
    # text  lemma   pos  tag  dep  shape  is_alpha  is_stop
    # accessed as token.text etc.

    ### DEBUG DEBUG DEBUG - start
    print(f"\n\nEach input sentence broken into words:\n")
    for idx, words_list in enumerate(sentences_words_list):
        print(f"Sentence {idx+1} :\n{words_list}")
    ### DEBUG DEBUG DEBUG - end

    ## create list of word tokens for each sentence doc WITHOUT STOP WORDS
    for idx, words_list in enumerate(sentences_words_list):
        for word in words_list:
            lexeme = nlp.vocab[word]
            if not lexeme.is_stop:
                sentences_words_list_no_stop[idx].append(word)
    
    ### DEBUG DEBUG DEBUG - start
    print(f"\n\nAfter removing any stop words:\n")
    for idx, words_list in enumerate(sentences_words_list_no_stop):
        print(f"Sentence {idx+1} :\n{words_list}")
    ### DEBUG DEBUG DEBUG - end

    ### DEBUG DEBUG DEBUG - start
    ## recreate the new doc array with the no stop words strings
    for words_list in sentences_words_list_no_stop:
        sentences_no_stop.append(' '.join(words_list))
    
    print(f"\n\nJoining the non-stop words as a new sentence:\n")
    for idx, new_sent_no_stop in enumerate(sentences_no_stop):
        print(f"New sentence {idx+1} :\n{new_sent_no_stop}")
    ### DEBUG DEBUG DEBUG - end
    
    # create data structure to hold all pos info for all all non-stop words
    pos_info = [ [] for _ in range(len(sentences_no_stop))]
    for idx1, words_list in enumerate(sentences_words_list_no_stop):
        for idx2, each_word in enumerate(words_list):
            pos_info[idx1].append({})
    
    # pos extraction and fill data structure
    pos_info = [ [] for _ in range(len(sentences_no_stop))]
    for idx1, words_list in enumerate(sentences_words_list_no_stop):
        for idx2, each_word in enumerate(words_list):
            pos_info[idx1].append({})
    for idx1, each_sent_no_stop in enumerate(sentences_no_stop):
        doc = nlp(each_sent_no_stop)
        for idx2, token in enumerate(doc):
            pos_info[idx1][idx2]['text']    = token.text
            pos_info[idx1][idx2]['lemma_']   = token.lemma_
            pos_info[idx1][idx2]['pos_']     = token.pos_
            pos_info[idx1][idx2]['tag_']     = token.tag_
            pos_info[idx1][idx2]['dep_']     = token.dep_
            pos_info[idx1][idx2]['shape_']   = token.shape_
            pos_info[idx1][idx2]['is_alpha'] = token.is_alpha
            pos_info[idx1][idx2]['is_stop']  = token.is_stop
    
    ### DEBUG DEBUG DEBUG - start
    #print(f"\n\nAll non-stop words pos info:\n")
    #for idx1, each_new_sent in enumerate(pos_info):
    #    print(f".........")
    #    for idx2, each_word_pos_info in enumerate(each_new_sent):
    #        print(f"Sentence {idx1+1}, word {idx2+1} :\n{each_word_pos_info}")
    
    #exit(0)
    ### DEBUG DEBUG DEBUG - end

    # write the file for all non-stop words pos info
    try:
        with open(opfileallposinfo, 'w') as opfileall:
                json.dump(pos_info, opfileall)
    except Exception as opfileallpos_write_error:
        print(f"\n\nFATAL ERROR: Problem creating the output file for all words pos info.\nError message: {opfileallpos_write_error}\nExiting with RC=500")
        exit(500)
    
    # extract the only the text of the noun type words from the pos info data structure
    #    then write it to output file
    sentences_only_nouns_list = []
    for sent_info in pos_info:
        each_sentence_nouns_list = []
        for each_word_info in sent_info:
            if each_word_info['tag_'] == 'NN':
                each_sentence_nouns_list.append(each_word_info['text'])
        # append to the final list only its not an empty list
        if each_sentence_nouns_list:
            sentences_only_nouns_list.append(each_sentence_nouns_list)
    
    ### DEBUG DEBUG DEBUG - start
    print(f"\n\nContents of list to be written to key elements file:\n{sentences_only_nouns_list}")
    ### DEBUG DEBUG DEBUG - end

    try:
        with open(opfilekeyelem, 'w') as opfilekeyonly:
                json.dump(sentences_only_nouns_list, opfilekeyonly)
    except Exception as opfilekeyonly_write_error:
        print(f"\n\nFATAL ERROR: Problem creating the output file for key elements.\nError message: {opfilekeyonly_write_error}\nExiting with RC=520")
        exit(520)


    print(f"\n\nNormal exit from program.\n")

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
