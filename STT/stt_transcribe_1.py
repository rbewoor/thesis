## -----------------------------------------------------------------------------
## Goal: Use Deepspeech model to run inference on the input wav files. Capture the output and save it to a file.
## -----------------------------------------------------------------------------
## Command line arguments:
## 1) -wavlocfile : location of input file with each line specifying the individual wav files to be processed.
## 2) -opfile     : location for the output file which will contain the inferences from deepspeech
## -----------------------------------------------------------------------------
## Usage example:
##    python3 stt_transcribe_1.py -wavlocfile "/home/rohit/PyWDUbuntu/thesis/SttTranscribe/stt_wav_files_loc_1.txt" -opfile "/home/rohit/PyWDUbuntu/thesis/SttTranscribe/stt_op_file_1.txt"
## -----------------------------------------------------------------------------

import argparse
import os
import subprocess

argparser = argparse.ArgumentParser(
    description='pick wav files specified in input file, process via deepspeech and capture transcription output')

argparser.add_argument(
    '-opfile',
    '--outputfile',
    help='location to save the output file which will contain the inference from deespeech model')

argparser.add_argument(
    '-wavlocfile',
    '--wavfilesloccationinputfile',
    help='file containing the location of all the wav files to process')

def stt_transcribe_functionality(args):
    # process command line arguments
    opfileloc    = args.outputfile      # -opfile parameter
    wavlocfile   = args.wavfilesloccationinputfile  # -wavlocfile parameter
    
    # check valid input for the -wavlocfile parameter
    # then check each of the wav files specified
    if not os.path.isfile(wavlocfile):
        print(f"FATAL ERROR: Input for wavlocfile parameter is not an existing file.\nExiting with RC=100")
        exit(100)
    else:
        try:
            wavfiles_arr = []
            with open(wavlocfile, 'r') as infile:
                for line in infile:
                    wavfiles_arr.append(line.rstrip("\n"))
            print(f"\n{wavfiles_arr}\n")
            for each_wav_file in wavfiles_arr:
                if not os.path.isfile(each_wav_file):
                    print(f"\n\nFATAL ERROR: Check the wav file locations specified in input file.\nThis is not a file:\n{each_wav_file}\nExiting with RC=200")
                    exit(200)
        except Exception as wavlocfile_read_error:
            print(f"\n\nFATAL ERROR: Problem reading the input file.\nError message: {wavlocfile_read_error}\nExiting with RC=300")
            exit(300)
    
    deepspeech_inferences_arr = []
    # create skeleton command
    ds_inf_cmd_fixed = "deepspeech " + \
                 "--model /home/rohit/deepspeech/pretrained/v073/deepspeech-0.7.3-models.pbmm " + \
                 "--scorer /home/rohit/deepspeech/pretrained/v073/deepspeech-0.7.3-models.scorer " + \
                 "--audio " #/home/rohit/PyWDUbuntu/thesis/audio/wavs/input1.wav - this last part will be added on the fly
    
    print(f"\n\nds_inf_cmd_fixed=\n{ds_inf_cmd_fixed}\n\n")

    for each_wav_file in wavfiles_arr: #[:1]:
        ds_inf_cmd = ds_inf_cmd_fixed + each_wav_file
        print(f"\nAbout to execute:::: {ds_inf_cmd}")
        inference_run = subprocess.Popen(ds_inf_cmd.split(' '), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = inference_run.communicate()
        inference_run.wait()
        deepspeech_inferences_arr.append(stdout.rstrip('\n'))
    
    print(f"\n\ndeepspeech_inferences_arr\n{deepspeech_inferences_arr}\n")
    
    # write the inferences to output file
    try:
        with open(opfileloc, 'w') as opfile:
            for each_inference in deepspeech_inferences_arr:
                opfile.write(each_inference + '.' + '\n')
    except Exception as opfile_write_error:
        print(f"\n\nFATAL ERROR: Problem creating the output file.\nError message: {opfile_write_error}\nExiting with RC=500")
        exit(500)
    
    print(f"\nOutput file created: {opfileloc}\n")
    print(f"\n\nNormal exit from program.\n")

if __name__ == '__main__':
    args = argparser.parse_args()
    stt_transcribe_functionality(args)
