## -----------------------------------------------------------------------------
## Goal:  Allow user to DESELECT images returned from the Neo4j query stage via a GUI. Reasons for this Deselection:
###             Reason 1) This is because some images may have wrongly found the objects of interest.
###             Reason 2) Each Neo4j query can return up to 20 images. But the maximum number of images per
###                       query to pass to the next stage (Auto caption block) is limited to 5. So allow culling.
###
###       Logic:
###           1) Show the up to 20 images in a grid pattern as thumbnails.
###           2) Provide option to enlarge image to study more clearly.
###           3) Provide option to toggle the selection button.
###              Track Deselections and show the count of currently Selected images to help user.
###           4) Once selections confirmed, make sure number of remaining images is within maximum limit
###                   that is currently 5.
###              If more than 5 were selected, then throw error message in the Root window and force
###                   user to start over again for that particular query.
###           5) When viewing an enlarged image, provide option to perform object detection inference again.
###                   Show a new window displaying the image with the bounding boxes for objects detected.
###       Outputs:
###          None
###              Only prints the before and after data structure to show that the images Deselected are
###                   are actually removed.
## -----------------------------------------------------------------------------
## PENDING: On the image inference window, show the textual output of model
## -----------------------------------------------------------------------------
## 
## Command line arguments: NONE
## -----------------------------------------------------------------------------
## Usage example:
##    python3 gui_query_neo_results_image_selection_1.py
## -----------------------------------------------------------------------------
##
## Layout info and tkinter variable
##
##     1) ROOT WINDOW: showing, instructions, warning, error message and button to start selection process for a query.
## 
##     o_root_window.root: tk.Root tpye: from class c_my_root_window
##     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
##     x                              Root title = "Root Window - Query number whatever"                                  x
##     x                                                                                                                  x
##     x       ----------------------------------------------------------------------------------------------             x
##     x       -                            Label: root.lbl_root_instructions                               -             x
##     x       ----------------------------------------------------------------------------------------------             x
##     x                                                                                                                  x
##     x               ------------------------------------------------------------------------                           x
##     x               -                    Label: root.lbl_root_error_message                -                           x
##     x               ------------------------------------------------------------------------                           x
##     x                                                                                                                  x
##     x               ------------------------------------------------------------------------                           x
##     x               -                Label: root.lbl_root_msg_warning_not_close            -                           x
##     x               ------------------------------------------------------------------------                           x
##     x                                                                                                                  x
##     x               ------------------------------------------------------------------------                           x
##     x               -                      Button: root.btn_root_click_proceed             -                           x
##     x               ------------------------------------------------------------------------                           x
##     x                                                                                                                  x
##     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## 
##
##     2) MAIN WINDOW: showing grid of resiyed images with options to Deselect/Select and Enlarge
## 
##     o_main_window.wnd: tk.Toplevel type: from class c_my_wnd_main_window
##     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
##     x                            Window title = "Thumbnail and Selection Window - Query number whatever"               x
##     x                                                                                                                  x
##     x                                                                                                                  x
##     x    o_main_window.frame_arr: is a 6 row x 10 columns = 60 grid cells. Alternating resized image Label,            x
##     x                             Enlarge Button, Selection button. The first grids in first 3 rows, all of            x
##     x                             first column are shown below.                                                        x
##     x                             Imagine this pattern repeats over the entire 60 grid cells.                          x
##     x    ------------------------------------------------------------------------------------------------------        x
##     x    -                                                                                                    -        x
##     x    -       ------------------------------------------------------------------------------               -        x
##     x    -       - Resized image: object of class my_frm_image, its Label: lbl_img            -               -        x
##     x    -       ------------------------------------------------------------------------------               -        x
##     x    -                                                                                                    -        x
##     x    -       -------------------------------------------------------------------------------              -        x
##     x    -       - "Enlarge": object of class my_frm_enlarge, its Button: btn_enlarge          -              -        x
##     x    -       -------------------------------------------------------------------------------              -        x
##     x    -                                                                                                    -        x
##     x    -       -------------------------------------------------------------------------------              -        x
##     x    -       - "Selected": object of class my_frm_select, its Button: btn_select           -              -        x
##     x    -       -------------------------------------------------------------------------------              -        x
##     x    -                                                                                                    -        x
##     x    ------------------------------------------------------------------------------------------------------        x
##     x                                                                                                                  x
##     x    ## grid has finished above, below Lable and Button span the entire width of window                            x
##     x                                                                                                                  x
##     x    ------------------------------------------------------------------------------------------------------        x
##     x    -                              Label: o_main_window.lbl_track_selected_count                         -        x
##     x    ------------------------------------------------------------------------------------------------------        x
##     x                                                                                                                  x
##     x    ------------------------------------------------------------------------------------------------------        x
##     x    -                                   Button: o_main_window.btn_done                                   -        x
##     x    ------------------------------------------------------------------------------------------------------        x
##     x                                                                                                                  x
##     x                                                                                                                  x
##     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## 
##
##     3) ENLARGE WINDOW: show particular image in original size and its location
## 
##     wnd_enlarged_img: tk.Toplevel type: from class my_frm_enlarge
##     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
##     x                                         Window title = "Enlarged image: image name"                              x
##     x                                                                                                                  x
##     x                                                                                                                  x
##     x                  -----------------------------------------------------------------------------                   x
##     x                  -                           Label: lbl_enlarged_img                         -                   x
##     x                  -----------------------------------------------------------------------------                   x
##     x                                                                                                                  x
##     x                  -----------------------------------------------------------------------------                   x
##     x                  -                           Label: lbl_enlarged_img_full_path               -                   x
##     x                  -----------------------------------------------------------------------------                   x
##     x                                                                                                                  x
##     x                  -----------------------------------------------------------------------------                   x
##     x                  -                        Button: btn_enlarged_img_do_inference              -                   x
##     x                  -----------------------------------------------------------------------------                   x
##     x                                                                                                                  x
##     x                                                                                                                  x
##     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## 
## 
##
##     4) INFERENCE WINDOW: show image with objects detected
## 
##     wnd_inference_img: tk.Toplevel type: from class my_frm_enlarge
##     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
##     x                                         Window title = "Inference image for: image name"                         x
##     x                                                                                                                  x
##     x                                                                                                                  x
##     x                  -----------------------------------------------------------------------------                   x
##     x                  -                           Label: lbl_inference_img                        -                   x
##     x                  -----------------------------------------------------------------------------                   x
##     x                                                                                                                  x
##     x                  -----------------------------------------------------------------------------                   x
##     x                  -                           Label: lbl_inference_of_this_image              -                   x
##     x                  -----------------------------------------------------------------------------                   x
##     x                                                                                                                  x
##     xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
## 
## 
## -----------------------------------------------------------------------------

# importing required packages
import tkinter as tk
from PIL import ImageTk, Image
import os
from functools import partial

from keras.models import Model as keras_Model, load_model as keras_load_model
import numpy as np
import struct
import cv2


#global SAVED_KERAS_MODEL_PATH = r'/home/rohit/PyWDUbuntu/thesis/saved_keras_model/yolov3_coco80.saved.model'

class c_my_root_window:
    def __init__(self, _query_num, _each_query_result, _func_show_grid_selection_window):
        self.query_num = _query_num
        self.each_query_result = _each_query_result
        self.msg_warning_for_root_do_not_close = r'DO NOT CLOSE THIS WINDOW TILL ALL SELECTIONS ARE COMPLETED FOR THIS QUERY!!!'
        self.msg_instructions_for_root = "".join([
            f"                               ----------------------------------",
            f"\n                               -----      INSTRUCTIONS      -----",
            f"\n                               ----------------------------------",
            f"\n",
            f"\n",
            f"\nThis is the main control window for a particular query.",
            f"\n",
            f"\n\t1.  Click on the button to proceed.",
            f"\n",
            f"\n\t2.  A grid displays for the candidate images from database query.",
            f"\n",
            f"\n\t3.  You can Deselect any images you want by toggling the Select button.",
            f"\n",
            f"\n\t4.  You can also Deselect all images.",
            f"\n",
            f"\n\t5.  But you can only Select maximum 5 images.",
            f"\n",
            f"\n\t6.  Once ready, you can click the button to Confirm Deselections.",
            f"\n",
            f"\n\t7.  Monitor the count of currently Selected images before confirming Deselections.",
            f"\n",
            f"\n\t8.  You can Enlarge an image to inspect it more cloself before deciding to Select or Deselect.",
            f"\n",
            f"\n\t9.  When viewing the Enlarged image, you can also perform image inference to show an"
            f"\n\t         Inference image in a new window with the objects detected.",
            f"\n",
            f"\n\t10.  Important: If you select more than 5 images and confirm Deselections,",
            f"\n\t                you will have to start the whole process of selection again.",
            f"\n",
            f"\n\t                If you accidentally close this window, the next query selection",
            f"\n\t                will start off."
        ])
        ## to hold deselected positions info. Eventually the inner list will
        ##    be replaced by an interget list of positions
        self.done_button_deselected_positions_results = [[]]
        self.func_show_grid_selection_window = _func_show_grid_selection_window

        self.root = tk.Tk()
        self.root.title(f"Root Window - Query number {_query_num}")
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=35,
            relief=tk.SUNKEN
            )
        self.lbl_root_error_message = tk.Label(
            master=self.root,
            text="No Errors detected so far.",
            bg="green", fg="white",
            justify=tk.LEFT,
            relief=tk.SUNKEN
            )
        self.lbl_root_error_message.configure(
            width=( len(self.lbl_root_error_message["text"]) + 60 ),
            height=3
            )
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to proceed to selection for Query {_query_num}",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=10,
            command=partial(
                self.func_show_grid_selection_window,
                self.root,
                self.each_query_result,
                self.query_num,
                self.done_button_deselected_positions_results,
                self.lbl_root_error_message
                )
            )
        self.btn_root_click_proceed.configure(
            width=100,
            height=7
        )
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.lbl_root_error_message.pack(padx=10, pady=10)
        self.lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
        self.btn_root_click_proceed.pack(padx=50, pady=50)

class c_my_wnd_main_window:
    def __init__(self, _root, _query_num, _root_done_button_deselected_positions_results, _num_candidate_images_in_this_query, _lbl_root_error_message, _o_keras_inference_performer):
        self.wnd = tk.Toplevel(master=_root)
        self.root = _root
        self.query_num = _query_num
        self.frame_arr = None
        self.frame_arr_n_rows = None
        self.frame_arr_n_cols = None
        self.o_keras_inference_performer = _o_keras_inference_performer
        self.max_limit_images_remaining_after_deselections = 5  ## after user makes deselections, the number of candidate images cannot be more than this limit
        self._num_candidate_images_in_this_query_at_start = _num_candidate_images_in_this_query
        self.lbl_root_error_message = _lbl_root_error_message
        self._root_done_button_deselected_positions_results = _root_done_button_deselected_positions_results
        self.frm_done = tk.Frame(
            master=self.wnd,
            relief=tk.FLAT,
            borderwidth=10
            )
        self.btn_done = tk.Button(master=self.frm_done, text="Click to Confirm Deselections",
            bg="yellow", fg="black",
            borderwidth=10,
            relief=tk.RAISED,
            command=self.do_button_done_processing
            )
        self.btn_done.configure(
            width= ( len(self.btn_done["text"]) + 20 ),
            height=5
            )
        self.frm_track_selected_count = tk.Frame(
                    master=self.wnd,
                    relief=tk.FLAT,
                    borderwidth=10
                    )
        self.lbl_track_selected_count = tk.Label(
            master=self.frm_track_selected_count,
            text=" ".join( [ "Count of Images currently Selected =", str(self._num_candidate_images_in_this_query_at_start) ]),
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=10
            )
        self.lbl_track_selected_count.configure(
            width= ( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        if self._num_candidate_images_in_this_query_at_start > 5:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
        self.wnd.title(f"Thumbnails and Selection Window -- Query number {_query_num}")
    
    def do_button_done_processing(self):
        ## For the images that are Deselected, figure out the position and add position number
        ##     to the return list for action later.
        index_positions_of_deselected_result = []
        for r_idx in range(2, self.frame_arr_n_rows, 3):
            for c_idx in range(self.frame_arr_n_cols):
                if self.frame_arr[r_idx][c_idx].btn_select["text"] == "Deselected":
                    index_positions_of_deselected_result.append( ( (r_idx // 3) * self.frame_arr_n_cols ) + c_idx )
        #print(f"\nDeselected positions=\n{index_positions_of_deselected_result}")

        ## check the number of deselections meets requirements for logic:
        ##       maximum selections remaining is 5 or less.
        ##       Note: User can deselect ALL images i.e. No images are ok as per user.
        num_images_remaining_after_deselections = self._num_candidate_images_in_this_query_at_start - len(index_positions_of_deselected_result)
        if num_images_remaining_after_deselections > self.max_limit_images_remaining_after_deselections:
            ## problem: candidate images remaning greater than maximum limit
            print(f"\nPROBLEM: For {self.query_num}: After Deselections, number of images remaining = {num_images_remaining_after_deselections}, which is greater than the maximum limit allowed = {self.max_limit_images_remaining_after_deselections}\n")
            print(f"\nYou need to restart the selection process....\n")
            ## change the color and message in the Error message Label in the root
            msg_text_for_root_error_label = "".join([
                f"ERROR: After Deselections, number of images remaining = {num_images_remaining_after_deselections},",
                f"\n     which is greater than the maximum limit = {self.max_limit_images_remaining_after_deselections}",
                f"\nYou need to restart the selection process...."
                ])
            self.lbl_root_error_message["text"] = msg_text_for_root_error_label
            self.lbl_root_error_message.configure(bg="red", fg="white", height=8, width=80)
            self.lbl_root_error_message.pack(padx=20, pady=20)
            self.wnd.destroy()
        else:
            ## all good so proceed to return information and destroy the root window
            ## complicated way to return values
            self._root_done_button_deselected_positions_results[0] = index_positions_of_deselected_result
            self.root.destroy()
    
    def build_frames(self, _n_rows, _n_cols, _in_img_arr_resized):
        ## the grid will always accomodate 20 images. The incoming array has tuple of the resized image and the
        ##     absolute path to image. But if the incoming array has less images, these entries will be None.

        ## create array holding skeleton form of the objects of required frame types
        self.frame_arr = []
        self.frame_arr_n_rows = _n_rows
        self.frame_arr_n_cols = _n_cols
        for r_idx in range(_n_rows):
            self.wnd.columnconfigure(r_idx, weight=1, minsize=50)
            self.wnd.rowconfigure(r_idx, weight=1, minsize=50)
            temp_array = []
            for c_idx in range(_n_cols):
                frame_name_str = ''.join( ['frm_', str(r_idx+1), '_', str(c_idx+1)] )
                if (r_idx % 3) == 0: ## picture frame
                    temp_array.append(my_frm_image(self.wnd, frame_name_str))
                elif (r_idx % 3) == 1: ## enlarge frame
                    temp_array.append(my_frm_enlarge(self.wnd, frame_name_str, self.o_keras_inference_performer))
                else: ## select frame
                    temp_array.append(my_frm_select(self.wnd, frame_name_str, self.lbl_track_selected_count))
            self.frame_arr.append(temp_array)
        
        ## set the frame arrays objects as required
        img_idx = 0
        for r_idx in range(_n_rows):
            for c_idx in range(_n_cols):
                if (r_idx) % 3 == 0: ## picture frame
                    self.frame_arr[r_idx][c_idx].frm_img.grid(row=r_idx,
                                                              column=c_idx,
                                                              padx=5, pady=5
                                                             )
                    ## populate the image
                    resized_image, file_with_path = _in_img_arr_resized[img_idx]
                    if resized_image is not None:
                        self.frame_arr[r_idx][c_idx].add_image(resized_image)
                        img_idx += 1
                        self.frame_arr[r_idx][c_idx].image_with_path = file_with_path
                    else:
                        self.frame_arr[r_idx][c_idx].image_with_path = None
                        self.frame_arr[r_idx][c_idx].lbl_img = tk.Label(
                            master=self.frame_arr[r_idx][c_idx].frm_img,
                            text=f"No Image",
                            bg="black", fg="white",
                            width=12, height=4
                            )
                        self.frame_arr[r_idx][c_idx].lbl_img.pack(padx=5, pady=5)
                elif (r_idx % 3) == 1: ## enlarge frame
                    self.frame_arr[r_idx][c_idx].frm_enlarge.grid(row=r_idx,
                                                              column=c_idx,
                                                              padx=5, pady=5
                                                             )
                    self.frame_arr[r_idx][c_idx].enlarge_this_image = self.frame_arr[r_idx - 1][c_idx].image_with_path ## bcoz immediately previous row should reference the image frame
                    ## disable the Enlarge button if there is no associated image
                    if self.frame_arr[r_idx][c_idx].enlarge_this_image is None:
                        self.frame_arr[r_idx][c_idx].btn_enlarge.configure(state=tk.DISABLED)
                    self.frame_arr[r_idx][c_idx].btn_enlarge.pack(padx=5, pady=5)
                else: ## select frame
                    self.frame_arr[r_idx][c_idx].frm_select.grid(row=r_idx,
                                                              column=c_idx,
                                                              padx=5, pady=5
                                                             )
                    ## disable the Select button if there is no associated image
                    if self.frame_arr[r_idx-2][c_idx].image_with_path is None:
                        self.frame_arr[r_idx][c_idx].btn_select.configure(state=tk.DISABLED)
                    self.frame_arr[r_idx][c_idx].btn_select.pack(padx=5, pady=5)
        
        ## the final button for submitting selected images
        r_idx = _n_rows
        c_idx = 0
        #self.wnd.columnconfigure(r_idx, weight=1, minsize=50)
        #self.wnd.rowconfigure(r_idx, weight=1, minsize=50)
        self.frm_done.grid(row=r_idx, column=c_idx, rowspan=1, columnspan=_n_cols, sticky="nsew", padx=5, pady=5)
        self.btn_done.pack(padx=10, pady=10, fill='both', expand=True)

        ## label to show how many images are currently selected
        r_idx = _n_rows + 1
        c_idx = 0
        #self.wnd.columnconfigure(r_idx, weight=1, minsize=150)
        #self.wnd.rowconfigure(r_idx, weight=1, minsize=150)
        self.frm_track_selected_count.grid(row=r_idx, column=c_idx, rowspan=1, columnspan=_n_cols, sticky="nsew", padx=5, pady=5)
        self.lbl_track_selected_count.pack(padx=10, pady=10, fill='both', expand=True)

class my_frm_image:
    def __init__(self, _wnd, _frame_name):
        
        self.frm_img = tk.Frame(
            master=_wnd,
            relief=tk.SUNKEN,
            borderwidth=2
            )
        self.frame_name = _frame_name
        self.image_with_path = None
        self.lbl_img = tk.Label(master=self.frm_img, image=None)
    
    def add_image(self, _in_img):
        self.lbl_img.configure(image=_in_img)
        self.lbl_img.pack(padx=1, pady=1)

class my_frm_enlarge:
    def __init__(self, _wnd, _frame_name, _o_keras_inference_performer):
        self.o_keras_inference_performer = _o_keras_inference_performer
        self.frm_enlarge = tk.Frame(
            master=_wnd,
            relief=tk.RAISED,
            borderwidth=4
            )
        self.master = _wnd
        self.frame_name = _frame_name
        self.enlarge_this_image = None
        self.btn_enlarge = tk.Button(
            master=self.frm_enlarge,
            text=f"Enlarge",
            bg="black", fg="white", 
            command=self.do_enlarge_btn_press_functionality
            )
        self.btn_enlarge.pack(padx=5, pady=5)
        ## enlarged image related - set as None at start
        self.wnd_enlarged_img = None
        self.lbl_enlarged_img = None
        self.lbl_enlarged_img_full_path = None
        self.btn_enlarged_img_do_inference = None
        ## inference image related - set as None at start
        self.wnd_inference_img = None
        self.lbl_inference_img = None
        self.lbl_inference_of_this_image = None
    
    def do_inference_and_display_output(self):
        print(f"\n\nInference invoked for: {self.enlarge_this_image}")
        self.wnd_inference_img = tk.Toplevel(master=self.wnd_enlarged_img)
        self.wnd_inference_img.title(f"Inference for: {os.path.basename(self.enlarge_this_image)}")
        self.lbl_inference_img = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=2,
            image=None
            )
        
        ## do the actual inference by saved keras model, then superimpose the bounding boxes and write the
        ##    image to an intermediate file
        self.o_keras_inference_performer.perform_model_inference(self.enlarge_this_image)

        ## reload the intermediate file for tkinter to display
        inference_model_output_image_path = r'./intermediate_file_inferenece_image.jpg'
        inference_model_output_image = ImageTk.PhotoImage(Image.open(inference_model_output_image_path))
        self.lbl_inference_img.configure(
            image=inference_model_output_image
            )
        self.lbl_inference_of_this_image = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image full path = {self.enlarge_this_image}"
            )
        self.lbl_inference_of_this_image.configure(
            width=( len(self.lbl_inference_of_this_image["text"]) + 10),
            height=3
            )
        self.lbl_inference_img.pack(padx=15, pady=15)
        self.lbl_inference_of_this_image.pack(padx=15, pady=15)
        self.wnd_inference_img.mainloop()

    def do_enlarge_btn_press_functionality(self):
        #print(f"Would have enlarged image: {self.enlarge_this_image}")
        img_orig = ImageTk.PhotoImage(Image.open(self.enlarge_this_image))
        self.wnd_enlarged_img = tk.Toplevel(master=self.master)
        self.wnd_enlarged_img.title(f"Enlarged image: {os.path.basename(self.enlarge_this_image)}")
        self.lbl_enlarged_img = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.SUNKEN,
            borderwidth=10,
            image=img_orig)
        self.lbl_enlarged_img_full_path = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image full path = {self.enlarge_this_image}"
            )
        self.lbl_enlarged_img_full_path.configure(
            width=( len(self.lbl_enlarged_img_full_path["text"]) + 10),
            height=3
            )
        self.btn_enlarged_img_do_inference = tk.Button(
            master=self.wnd_enlarged_img,
            relief=tk.RAISED,
            borderwidth=10,
            text="Click to run Inference and show images with detected objects",
            bg="yellow", fg="black",
            command=self.do_inference_and_display_output
            )
        self.btn_enlarged_img_do_inference.configure(
            width=( len(self.btn_enlarged_img_do_inference["text"]) + 10 ),
            height=3
            )
        self.lbl_enlarged_img.pack(padx=15, pady=15)
        self.lbl_enlarged_img_full_path.pack(padx=15, pady=15)
        self.btn_enlarged_img_do_inference.pack(padx=15, pady=15)
        self.wnd_enlarged_img.mainloop()

class c_keras_inference_performer:
    
    def __init__(self):
        self.reloaded_yolo_model = None
    
    class BoundBox:
        def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
            self.xmin = xmin
            self.ymin = ymin
            self.xmax = xmax
            self.ymax = ymax
            
            self.objness = objness
            self.classes = classes

            self.label = -1
            self.score = -1
        
        def get_label(self):
            if self.label == -1:
                self.label = np.argmax(self.classes)
            return self.label
        
        def get_score(self):
            if self.score == -1:
                self.score = self.classes[self.get_label()]  
            return self.score
    
    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b

        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3
    
    def _sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
        
        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
        w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
        
        union = w1*h1 + w2*h2 - intersect
        
        return float(intersect) / union
    
    def preprocess_input(self, image, net_h, net_w):
        new_h, new_w, _ = image.shape

        # determine the new size of the image
        if (float(net_w)/new_w) < (float(net_h)/new_h):
            new_h = (new_h * net_w)/new_w
            new_w = net_w
        else:
            new_w = (new_w * net_h)/new_h
            new_h = net_h
        
        # resize the image to the new size
        resized = cv2.resize(image[:,:,::-1]/255., (int(new_w), int(new_h)))

        # embed the image into the standard letter box
        new_image = np.ones((net_h, net_w, 3)) * 0.5
        new_image[int((net_h-new_h)//2):int((net_h+new_h)//2), int((net_w-new_w)//2):int((net_w+new_w)//2), :] = resized
        new_image = np.expand_dims(new_image, 0)

        return new_image
    
    def decode_netout(self, netout, anchors, obj_thresh, nms_thresh, net_h, net_w):
        grid_h, grid_w = netout.shape[:2]
        nb_box = 3
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5

        boxes = []

        netout[..., :2]  = self._sigmoid(netout[..., :2])
        netout[..., 4:]  = self._sigmoid(netout[..., 4:])
        netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh

        for i in range(grid_h*grid_w):
            row = i / grid_w
            col = i % grid_w
            
            for b in range(nb_box):
                # 4th element is objectness score
                objectness = netout[int(row)][int(col)][b][4]
                #objectness = netout[..., :4]
                
                if(objectness.all() <= obj_thresh): continue
                
                # first 4 elements are x, y, w, and h
                x, y, w, h = netout[int(row)][int(col)][b][:4]

                x = (col + x) / grid_w # center position, unit: image width
                y = (row + y) / grid_h # center position, unit: image height
                w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
                h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
                
                # last elements are class probabilities
                classes = netout[int(row)][col][b][5:]
                
                box = self.BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

                boxes.append(box)
        
        return boxes
    
    def correct_yolo_boxes(self, boxes, image_h, image_w, net_h, net_w):
        if (float(net_w)/image_w) < (float(net_h)/image_h):
            new_w = net_w
            new_h = (image_h*net_w)/image_w
        else:
            new_h = net_w
            new_w = (image_w*net_h)/image_h
        
        for i in range(len(boxes)):
            x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
            y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
            
            boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
            boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
            boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
            boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
    
    def do_nms(self, boxes, nms_thresh):
        if len(boxes) > 0:
            nb_class = len(boxes[0].classes)
        else:
            return
        
        for c in range(nb_class):
            sorted_indices = np.argsort([-box.classes[c] for box in boxes])

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if boxes[index_i].classes[c] == 0: continue

                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                        boxes[index_j].classes[c] = 0
    
    def draw_boxes(self, image, boxes, labels, obj_thresh):
        for box in boxes:
            label_str = ''
            label = -1
            
            for i in range(len(labels)):
                if box.classes[i] > obj_thresh:
                    label_str += labels[i]
                    label = i
                    print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
            
            if label >= 0:
                cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
                cv2.putText(image, 
                            label_str + ' ' + str(box.get_score()), 
                            (box.xmin, box.ymin - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1e-3 * image.shape[0], 
                            (0,255,0), 2)
        return image
        
    def perform_model_inference(self, _path_infer_this_image):
        print(f"\n\nExecuting model inference on image_to_infer : {_path_infer_this_image}\n")
        
        ## load the keras pretained model if not already been loaded earlier
        if self.reloaded_yolo_model is None:
            print(f"\n\n    LOADED KERAS MODEL      \n\n")
            saved_model_location = r'/home/rohit/PyWDUbuntu/thesis/saved_keras_model/yolov3_coco80.saved.model'
            self.reloaded_yolo_model = keras_load_model(saved_model_location)

        ## set some parameters for network
        net_h, net_w = 416, 416
        obj_thresh, nms_thresh = 0.5, 0.45
        anchors = [[116,90,  156,198,  373,326],  [30,61, 62,45,  59,119], [10,13,  16,30,  33,23]]
        labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", \
                "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", \
                "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", \
                "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", \
                "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", \
                "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", \
                "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", \
                "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", \
                "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", \
                "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
        
        image_to_infer_cv2 = cv2.imread(_path_infer_this_image)
        image_h, image_w, _ = image_to_infer_cv2.shape
        try:
            image_to_infer_preprocessed = self.preprocess_input(image_to_infer_cv2, net_h, net_w)
        except Exception as error_inference_preprocess_image:
            print(f"\nFATAL ERROR: Problem reading the input file.\nError message: {error_inference_preprocess_image}\nExit RC=400")
            exit(400)
        
        ## run the prediction
        yolos = self.reloaded_yolo_model.predict(image_to_infer_preprocessed)
        boxes = []

        for i in range(len(yolos)):
            ## decode the output of the network
            boxes += self.decode_netout(yolos[i][0], anchors[i], obj_thresh, nms_thresh, net_h, net_w)
        
        ## correct the sizes of the bounding boxes
        self.correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)

        ## suppress non-maximal boxes
        self.do_nms(boxes, nms_thresh)

        ## draw bounding boxes into the image
        self.draw_boxes(image_to_infer_cv2, boxes, labels, obj_thresh)

        ## save the image as intermediate file -- see later whether to return and processing is possible
        cv2.imwrite(r'./intermediate_file_inferenece_image.jpg', (image_to_infer_cv2).astype('uint8') )

class my_frm_select:
    def __init__(self, _wnd, _frame_name, _lbl_track_selected_count):
        self._lbl_track_selected_count = _lbl_track_selected_count
        self.frm_select = tk.Frame(
            master=_wnd,
            relief=tk.RAISED,
            borderwidth=4)
        self.frame_name = _frame_name
        self.btn_select = tk.Button(
            master=self.frm_select,
            text=f"Selected",
            bg="black", fg="white",
            command=self.do_select_btn_press_functionality
        )
        self.btn_select.pack(padx=5, pady=5)
    
    def do_select_btn_press_functionality(self):
        #print(f"Pressed Select button: {self.frame_name}")
        ## get the current count
        current_count_selected = int( self._lbl_track_selected_count["text"].split()[-1] )
        fixed_message = " ".join( self._lbl_track_selected_count["text"].split()[:-1] )
        if self.btn_select["text"] == 'Selected':
            ## Sink button and change text to Deselect
            self.btn_select.configure(text=f"Deselected", relief=tk.SUNKEN, bg="yellow", fg="black")
            current_count_selected -= 1
        else:
            ## Raise button and change text to Select
           self.btn_select.configure(text=f"Selected", relief=tk.RAISED, bg="black", fg="white")
           current_count_selected += 1
        self._lbl_track_selected_count["text"] = " ".join( [fixed_message, str(current_count_selected)] )
        if current_count_selected > 5:
            self._lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self._lbl_track_selected_count.configure(bg="green", fg="white")
        return

def show_grid_selection_window(_root, _each_query_result, _query_num, _root_done_button_deselected_positions_results, _lbl_root_error_message):
    ## dictionary with key as the image source, value is the location of that datasets images
    source_and_location_image_datasets = {
        'flickr30k' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/flickr30k_images/flickr30k_images/',
        'coco_val_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_val2017_5k/val2017/',
        'coco_test_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_test2017_41k/test2017/',
        'coco_train_2017' : r'/media/rohit/DATA/EverythingD/01SRH-BDBA Acads/Thesis/StoryGenerator/Data/COCO_train2017_118k/'
        }
    
    ## build the list for images with their full path
    image_files_list = []
    for each_img_info in _each_query_result:
        image_files_list.append(
            os.path.join(
                source_and_location_image_datasets[each_img_info["Source"]],
                each_img_info["Image"]
                ))
    
    num_candidate_images_in_this_query = len(image_files_list)
    #print(f"Num of images = {num_candidate_images_in_this_query}\narray=\n{image_files_list}")

    opened_imgs_arr = []
    for idx in range(20):
        if idx < num_candidate_images_in_this_query:  ## image list has a valid entry to pick up
            img_orig = Image.open(image_files_list[idx])
            img_resized = img_orig.resize((100, 80),Image.ANTIALIAS)
            ## append tuple of resized image and file with path
            opened_imgs_arr.append( (ImageTk.PhotoImage(img_resized), image_files_list[idx]) )
            del img_orig, img_resized
        else:  ## there is no image
            opened_imgs_arr.append( (None, None) )
    
    ## the selection grid can handle maximum 20 image thumbnails.
    ##     There are 10 columns.
    ##     There are 2 rows of thumbnail images.
    ##     Each row of image has associate Enlarge row and Select row i.e. three rows 
    ##          are logically are part of each image row.
    #      So the selection window grid will have have 2 x 3 x 10 = 60 frames.
    n_rows = 2 * 3
    n_cols = 10
    #print(f"n_rows = {n_rows}\t\tn_cols = {n_cols}\n")

    o_keras_inference_performer = c_keras_inference_performer()
    o_main_window = c_my_wnd_main_window(_root, _query_num, _root_done_button_deselected_positions_results, num_candidate_images_in_this_query, _lbl_root_error_message, o_keras_inference_performer)
    o_main_window.build_frames(n_rows, n_cols, opened_imgs_arr)
    o_main_window.wnd.mainloop()

def select_images_functionality_for_one_query_result(_DEBUG_SWITCH, _query_num, _each_query_result):
    
    ## make a root window and show it
    o_root_window = c_my_root_window(_query_num, _each_query_result, show_grid_selection_window)
    #_query_num, _each_query_result, _func_show_grid_selection_window
    o_root_window.root.mainloop()
    o_root_window.done_button_deselected_positions_results
    if _DEBUG_SWITCH:
        print(f"\nQuery {_query_num} :::  o_root_window.done_button_deselected_positions_results = {o_root_window.done_button_deselected_positions_results}")
    return o_root_window.done_button_deselected_positions_results[0]

def selection_processing_functionality(_DEBUG_SWITCH):
    database_query_results = [
        [
            {'Image': '000000169542.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000169516.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000313777.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000449668.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000292186.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000168815.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000168743.jpg', 'Source': 'coco_test_2017'}
            ],
        [
            {'Image': '000000146747.jpg', 'Source': 'coco_test_2017'},
            {'Image': '000000509771.jpg', 'Source': 'coco_test_2017'}
            ],
        [
            {'Image': '000000012149.jpg', 'Source': 'coco_test_2017'}
            ]
        ]
    
    final_remaining_images_selected_info = []
    index_positions_to_remove_all_queries = []
    for query_num, each_query_result in enumerate(database_query_results):
        temp_array = []
        if _DEBUG_SWITCH:
            print(f"\n\nStarting Selection process for Query {query_num + 1}")
            num_candidate_images_before_selection_began = len(each_query_result)
            print(f"Number of images before selection began = {num_candidate_images_before_selection_began}\n")
        index_positions_to_remove_this_query = select_images_functionality_for_one_query_result(_DEBUG_SWITCH, query_num + 1, each_query_result)
        index_positions_to_remove_all_queries.append(index_positions_to_remove_this_query)
        num_images_to_remove = len(index_positions_to_remove_this_query)
        if _DEBUG_SWITCH:
            print(f"\nNumber of images Deselected by user = {num_images_to_remove}.\nNumber of images that will remain = { num_candidate_images_before_selection_began - num_images_to_remove }")
        ## remove the Deselected images
        for idx_each_image, each_image_info in enumerate(each_query_result):
            if idx_each_image not in index_positions_to_remove_this_query:
                temp_array.append(each_image_info)
        final_remaining_images_selected_info.append(temp_array)
        if _DEBUG_SWITCH:
            print(f"\nCompleted selection process - Query number {query_num + 1}\n")
    
    ## show summary info
    print(f"\n\n-------------------------------- SUMMARY INFORMATON --------------------------------")
    for query_num, (each_query_results, each_query_final_remaining_images_info, each_query_index_positions_remove) in \
        enumerate(zip(database_query_results, final_remaining_images_selected_info, index_positions_to_remove_all_queries)):
        print(f"For Query {query_num + 1}\nNumber of candidate images before selection = {len(each_query_results)}")
        print(f"Number of Deselections done = {len(each_query_index_positions_remove)}")
        print(f"Number of images remaining after Deselections = {len(each_query_final_remaining_images_info)}")
        if _DEBUG_SWITCH:
            print(f"\n\t------ Query images info BEFORE::\n{each_query_results}")
            print(f"\n\t------ Positions removed::\n{each_query_index_positions_remove}")
            print(f"\n\t------ Query images info AFTER::\n{each_query_final_remaining_images_info}\n\n")
    
    print(f"\n\nNormal exit.\n\n")

if __name__ == '__main__':
    DEBUG_SWITCH = True
    selection_processing_functionality(DEBUG_SWITCH)
    