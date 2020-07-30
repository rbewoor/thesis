## -----------------------------------------------------------------------------
## Goal:  Allow user to DESELECT images returned from the Neo4j query stage via a GUI. Reasons for this Deselection:
###             Reason 1) Object detection stage wrongly found objects of interest in the image
###             Reason 2) Each Neo4j query can return up to 20 images. But the maximum number of images per
###                       query to pass to the next stage (Auto caption block) is limited to 5. So allow culling.
###
###       Logic:
###           1) Show the up to 20 images in a grid pattern as thumbnails.
###           2) Provide option to enlarge each image to study more clearly - dedicated Enlarge Button.
###           3) Provide option to Select or Deselect an image by clicking its thumbnail.
###           4) Show live count of Deselected images to help user. Change color if current count greater than max. limit.
###           4) Once Deselections confirmed, ensure total Selections is within maximum limit (currently 5).
###              Disable the confirmation button if the current selection count is greater than max. limit.
###           5) When viewing an enlarged image, provide option to perform object detection inference again.
###                   Show a new window displaying the image with the bounding boxes for objects detected.
###       Outputs:
###          None
###              Only prints the before and after data structure to show that the images Deselected are
###                   are actually removed.
## -----------------------------------------------------------------------------
## Command line arguments: NONE
## -----------------------------------------------------------------------------
## Usage example:
##    python3 gui_query_neo_results_image_selection_9.py
## -----------------------------------------------------------------------------
##

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

class c_queryImgSelection_root_window:
    def __init__(self, _DEBUG_SWITCH, _query_num, _each_query_result, _index_positions_to_remove_this_query, _num_candidate_images_before_selection_began):
        self.DEBUG_SWITCH = _DEBUG_SWITCH
        self.query_num = _query_num
        self.each_query_result = _each_query_result
        ## this is the list passed from beginning. to be populated with positions of deselections during the
        ##      grid window processing
        self.index_positions_to_remove_this_query = _index_positions_to_remove_this_query
        
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
            f"\n\t2.  A grid displays thumbnails of the candidate images from database query.",
            f"\n\t       Important: If there are no images to display you cannot click to proceed to the"
            f"\n\t                  grid selection window. Simply close this window and the next query"
            f"\n\t                  selections process will begin."
            f"\n",
            f"\n\t3.  By default, all the images are Selected.",
            f"\n\t       You can Deselect any images by clicking the image thumbnail.",
            f"\n\t       You can also Deselect all images.",
            f"\n\t       But you can only Select a maximum of 5 images.",
            f"\n\t       Monitor the count of currently Selected images.",
            f"\n",
            f"\n\t4.  Once ready with your Selections, click the button to Confirm Deselections.", 
            f"\n\t       NOTE: While the number of Selections is invalid (more than 5), you cannot",
            f"\n\t             click the button to confirm Selections.",
            f"\n",
            f"\n\t5.  You can Enlarge an image to inspect it more cloself before deciding to Select or Deselect.",
            f"\n",
            f"\n\t6.  When viewing the Enlarged image, you can also perform object detection on the image to",
            f"\n\t         see an Inference image in a new window."
        ])

        ## root window for the query being processed
        self.root = tk.Tk()
        if _num_candidate_images_before_selection_began == 0:
            self.root.title(f"Image Selection - Query number {_query_num} - No Images to Display -- Please close this window")
        else:
            self.root.title(f"Image Selection - Query number {_query_num} - Click button to Proceed")
        ## label for warning message not to close
        self.lbl_root_msg_warning_not_close = tk.Label(
            master=self.root,
            text=self.msg_warning_for_root_do_not_close,
            bg="red", fg="white",
            width=(len(self.msg_warning_for_root_do_not_close) + 10),
            height=5
            )
        ## label for instructions
        self.lbl_root_instructions = tk.Label(
            master=self.root,
            text=self.msg_instructions_for_root,
            bg="blue", fg="white",
            justify=tk.LEFT,
            width=130,
            height=35,
            relief=tk.SUNKEN
            )
        ## button to proceed to grid selection window
        ## assume there are images and make proceed button clickable
        self.btn_root_click_proceed = tk.Button(
            master=self.root,
            text=f"Click to view images and make Selections - Query {_query_num}",
            bg="yellow", fg="black",
            relief=tk.RAISED,
            borderwidth=15,
            command=partial(
                generic_show_grid_selection_window,
                self.root,
                self.each_query_result,
                self.query_num,
                self.index_positions_to_remove_this_query
                )
            )
        ## if no images to display in grid, disable the button to proceed and change the text displayed in button
        if _num_candidate_images_before_selection_began == 0:
            self.btn_root_click_proceed.configure(
            state=tk.DISABLED,
            relief=tk.FLAT,
            text=f"No images to make Selections - Query {_query_num} - Please close this window",
            )
        self.btn_root_click_proceed.configure(
            width=(len(self.btn_root_click_proceed["text"]) + 10),
            height=7
        )
        self.lbl_root_msg_warning_not_close.pack(padx=15, pady=15)
        self.lbl_root_instructions.pack(padx=10, pady=10)
        self.btn_root_click_proceed.pack(padx=50, pady=50)

def generic_show_grid_selection_window(_root, _each_query_result, _query_num, _index_positions_to_remove_this_query):
    
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
    ##     Each row of images has associated Enlarge buttons row below it.
    ##     So the selection window grid will have have 2 x 2 x 10 = 40 buttons.
    ##     In addition to these, there are two more tkinter widgets:
    ##        a Button for Confirm Selections, and
    ##        a Label to show live count of currently Selected images.
    ##     NOTE: We specify the number of rows and columns wrt the images. Logic will later assume
    ##           two additional rows for the Confirm button and Selection count Label.
    n_rows_images = 2
    n_cols_images = 10
    n_rows = 2 * 2 ## one for the images, one for the associated Enlarge button
    n_cols = n_cols_images

    ## make object for object detector - this same object will be used for inferennce on all images
    ##      in the grid. So no need to load new model for each inference.
    o_keras_inference_performer = c_keras_inference_performer()

    o_queryImgSelection_grid_wnd_window = c_queryImgSelection_grid_wnd_window(
        _root,
        _query_num,
        _index_positions_to_remove_this_query,
        num_candidate_images_in_this_query,
        n_rows,
        n_cols,
        opened_imgs_arr,  ## REMEMBER this is tuple of (Photo image object resized, image path)
        o_keras_inference_performer
        )
    
    o_queryImgSelection_grid_wnd_window.wnd_grid.mainloop()

class c_queryImgSelection_grid_wnd_window:
    def __init__(
        self,
        _root,
        _query_num,
        _index_positions_to_remove_this_query,
        _num_candidate_images_in_this_query,
        _n_rows,
        _n_cols,
        _opened_imgs_arr,
        _o_keras_inference_performer
        ):

        self.root = _root
        self.query_num = _query_num
        self.index_positions_to_remove_this_query = _index_positions_to_remove_this_query
        self.num_candidate_images_in_this_query_at_start = _num_candidate_images_in_this_query
        self.n_rows = _n_rows   ## NOTE: This only counts the thumbnail and Enlarge rows, not for Confirm button and Label for current count
        self.n_cols = _n_cols
        self.opened_imgs_arr = _opened_imgs_arr
        self.o_keras_inference_performer = _o_keras_inference_performer
        
        ## array for the buttons for thumbnail image button and the associated Enlarge button
        self.grid_buttons_arr = []
        ## The number of Selected candidate images cannot be more than this limit
        self.max_limit_images_selections = 5
        ## initialise the current count of selections to the number of images at start
        self.images_selected_current_count = self.num_candidate_images_in_this_query_at_start

        ## window for the grid selection
        self.wnd_grid = tk.Toplevel(master=_root)
        self.wnd_grid.title(f"Thumbnails and Selection Window -- Query number {self.query_num}")
        ## label for count of current selections
        self.lbl_track_selected_count = tk.Label(
            master=self.wnd_grid,
            text="",
            relief=tk.FLAT,
            borderwidth=10
            )
        ## call function to update the text with count, and the color 
        self.update_label_count_currently_selected_images()
        ## button for selection confirm
        self.btn_confirm_selections = tk.Button(
            master=self.wnd_grid,
            text=f"Click to Confirm Deselections -- clickable only if current selection count is NOT > {self.max_limit_images_selections}",
            bg="yellow", fg="black",
            borderwidth=10,
            relief=tk.RAISED,
            command=self.do_confirm_selections_processing
            )
        self.btn_confirm_selections.configure(
            width=( len(self.btn_confirm_selections["text"]) + 20 ),
            height=5
            )
        ## prevent Confirm Selections if there are too many images Selected at start
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        
        ## populate the button array for the grid - thumbnail and Enlarge buttons
        ##    first make skeleton entries for the buttons
        ##    by default assume no Image is present to display,
        ##       so all buttons are in disabled state with the text saying "No Image"
        for r_idx in range(self.n_rows):
            self.wnd_grid.columnconfigure(r_idx, weight=1, minsize=50)
            self.wnd_grid.rowconfigure(r_idx, weight=1, minsize=50)
            temp_row_data = []
            for c_idx in range(self.n_cols):
                ## alternate row entries for button of image thumbnail and Enlarge
                if r_idx % 2 == 0:
                    ## thumbnail image button
                    temp_row_data.append(
                        tk.Button(
                            master=self.wnd_grid,
                            text="No Image",
                            bg="black", fg="white",
                            relief=tk.FLAT,
                            borderwidth=10,
                            state=tk.DISABLED
                            )
                        )
                else:
                    ## Enlarge button type
                    temp_row_data.append(
                        tk.Button(
                            master=self.wnd_grid,
                            text="Enlarge",
                            bg="black", fg="white",
                            relief=tk.FLAT,
                            borderwidth=10,
                            state=tk.DISABLED
                            )
                        )
            self.grid_buttons_arr.append(temp_row_data)

        ## now populate the Images and activate both button where applicable
        img_idx = 0 ## index to access each image from the tuple of (image, path)
        for r_idx in range(self.n_rows):
            for c_idx in range(self.n_cols):
                ## set grid position for all the label elements
                self.grid_buttons_arr[r_idx][c_idx].grid(
                    row=r_idx, column=c_idx,
                    padx=5, pady=5,
                    sticky="nsew"
                )
                ## only for the thumbnail rows, populate the images if it is available
                ##      if yes, change the state of thumbnail and associated Enlarge buttons
                if (r_idx % 2 == 0) and (img_idx < self.num_candidate_images_in_this_query_at_start):
                    ## r_idx is for an image thumbnail row and there is an image to show
                    ## from the input tuple extract the image and the path
                    resized_image, self.grid_buttons_arr[r_idx + 1][c_idx].image_path = self.opened_imgs_arr[img_idx]
                    self.grid_buttons_arr[r_idx][c_idx].image = None
                    self.grid_buttons_arr[r_idx][c_idx].configure(
                            image=resized_image,
                            relief=tk.SUNKEN,
                            borderwidth=10,
                            highlightthickness = 15,
                            highlightbackground = "green", highlightcolor= "green",
                            state=tk.NORMAL,
                            command=partial(
                                self.do_image_select_button_clicked_processing,
                                r_idx, c_idx
                                )
                        )
                    img_idx += 1
                    ## make variable to hold an Enlarged image window object for the associated Enlarge button.
                    ##     set as None for now.
                    ##     if associated Enlarge button is clicked, object will be populated and used.
                    self.grid_buttons_arr[r_idx + 1][c_idx].o_EnlargeImage_window = None
                    ## change the associated Enlarge button
                    self.grid_buttons_arr[r_idx + 1][c_idx].configure(
                            relief=tk.RAISED,
                            borderwidth=10,
                            state=tk.NORMAL,
                            command=partial(
                                generic_show_enlarged_image_window,
                                self.wnd_grid,
                                self.grid_buttons_arr[r_idx + 1][c_idx].o_EnlargeImage_window,
                                self.grid_buttons_arr[r_idx + 1][c_idx].image_path,
                                self.o_keras_inference_performer
                                )
                        )
        
        ## label for count of current selections
        r_idx = self.n_rows
        c_idx = 0
        self.lbl_track_selected_count.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols,
            sticky="nsew",
            padx=5, pady=5
            )

        ## button for selection confirm
        r_idx = self.n_rows + 1
        c_idx = 0
        self.btn_confirm_selections.grid(
            row=r_idx, column=c_idx,
            rowspan=1, columnspan=self.n_cols,
            sticky="nsew",
            padx=5, pady=5
            )
        
        return
    
    def do_confirm_selections_processing(self):
        ## For the images that are Deselected, figure out the position and add the position number
        ##     to the return list.
        self.index_positions_to_remove_this_query
        for r_idx in range(0, self.n_rows, 2):
            for c_idx in range(self.n_cols):
                if self.grid_buttons_arr[r_idx][c_idx]["relief"] == tk.RAISED:
                    ## this image is Deselected - so extract the position
                    self.index_positions_to_remove_this_query.append( ( (r_idx // 2) * self.n_cols ) + c_idx )
        print(f"\nFor Query {self.query_num}, Deselected positions=\n{self.index_positions_to_remove_this_query}")
        self.root.destroy()
        return

    def update_label_count_currently_selected_images(self):
        ## update the count based on latest count of selected images
        ##        also change the color if the count is greater than allowed limit
        self.lbl_track_selected_count.configure(
            text=" ".join([ "Count of Images currently Selected =", str(self.images_selected_current_count) ])
            )
        self.lbl_track_selected_count.configure(
            width=( len(self.lbl_track_selected_count["text"]) + 10 ),
            height=5
            )
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.lbl_track_selected_count.configure(bg="red", fg="white")
        else:
            self.lbl_track_selected_count.configure(bg="green", fg="white")
        return

    def do_image_select_button_clicked_processing(self, _r_idx, _c_idx):
        ## toggle button characteristics:
        ##                              Relief         Color around image
        ##       Selected   Image       SUNKEN         Green
        ##       Deselected Image       RAISED         Red
        if self.grid_buttons_arr[_r_idx][_c_idx]["relief"] == tk.SUNKEN:
            ## Image is currently Selected, change to Deselected
            self.grid_buttons_arr[_r_idx][_c_idx].configure(
                relief=tk.RAISED,
                highlightbackground = "red", highlightcolor= "red"
            )
            self.images_selected_current_count -= 1
        else:
            ## Image is currently Deselected, change to Selected
            self.grid_buttons_arr[_r_idx][_c_idx].configure(
                relief=tk.SUNKEN,
                highlightbackground = "green", highlightcolor= "green"
            )
            self.images_selected_current_count += 1
        ## update the label for count
        self.update_label_count_currently_selected_images()
        ## update the confirm button characteristics
        if self.images_selected_current_count > self.max_limit_images_selections:
            self.btn_confirm_selections.configure(state=tk.DISABLED, relief=tk.FLAT)
        else:
            self.btn_confirm_selections.configure(state=tk.NORMAL, relief=tk.RAISED)
        return

def generic_show_enlarged_image_window(_wnd_grid, _o_EnlargeImage_window, _image_path, _o_keras_inference_performer):
    ## make the object for Enlarged window and show the window. Currently object is None
    _o_EnlargeImage_window = c_EnlargeImage_window( _wnd_grid, _image_path, _o_keras_inference_performer)
    _o_EnlargeImage_window.wnd_enlarged_img.mainloop()

class c_EnlargeImage_window:
    def __init__(self, _master_wnd, _image_path, _o_keras_inference_performer):
        self.master = _master_wnd
        self.image_path = _image_path
        self.o_keras_inference_performer = _o_keras_inference_performer

        self.img_orig = ImageTk.PhotoImage(Image.open(self.image_path))

        ## window for the Enlarged image
        self.wnd_enlarged_img = tk.Toplevel(master=self.master)
        self.wnd_enlarged_img.title(f"Enlarged image: {os.path.basename(self.image_path)}")
        ## label for Enlarged image
        self.lbl_enlarged_img = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.FLAT,
            borderwidth=10,
            image=self.img_orig)
        self.lbl_enlarged_img_full_path = tk.Label(
            master=self.wnd_enlarged_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image path = {self.image_path}"
            )
        self.lbl_enlarged_img_full_path.configure(
            width=( len(self.lbl_enlarged_img_full_path["text"]) + 10),
            height=3
            )
        ## button for Inference
        self.btn_enlarged_img_do_inference = tk.Button(
            master=self.wnd_enlarged_img,
            relief=tk.RAISED,
            borderwidth=10,
            text="Click to perform object detection inference on this image",
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
    
    def do_inference_and_display_output(self):
        print(f"\n\nInference invoked for: {self.image_path}")

        ## window for inference
        self.wnd_inference_img = tk.Toplevel(master=self.wnd_enlarged_img)
        self.wnd_inference_img.title(f"Inference for: {os.path.basename(self.image_path)}")
        ## label for the output image after inference - set as None for now
        self.lbl_inference_img = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.FLAT,
            borderwidth=4,
            image=None
            )
        ## label for the image path
        self.lbl_path_img_inferred = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="grey", fg="white",
            text=f"Image path = {self.image_path}"
            )
        self.lbl_path_img_inferred.configure(
            width=( len(self.lbl_path_img_inferred["text"]) + 10),
            height=3
            )
        ## label for the inference info of types of objects found
        self.lbl_inference_textual_info = tk.Label(
            master=self.wnd_inference_img,
            relief=tk.SUNKEN,
            borderwidth=5,
            bg="blue", fg="white",
            justify='left'
            )
        ## do the actual inference by saved keras model, then superimpose the bounding boxes and write the
        ##    output image to an intermediate file.
        ## But first, make empty list to to be populated by model inference performer.
        ##       entries made will be f-string of name of objects found with percentage
        ##            without the new-line character at end.
        self.objects_found_info_arr = []
        self.o_keras_inference_performer.perform_model_inference(self.image_path, self.objects_found_info_arr)
        ## reload the intermediate file for tkinter to display
        inference_model_output_image_path = r'./intermediate_file_inferenece_image.jpg'
        inference_model_output_img = ImageTk.PhotoImage(Image.open(inference_model_output_image_path))
        ## put the inference outpu image in the label widget
        self.lbl_inference_img.configure(image=inference_model_output_img)
        ## extract the info about objects found and put into the label text
        textual_info_to_display = "\n".join( self.objects_found_info_arr )
        self.lbl_inference_textual_info.configure(text=textual_info_to_display)
        self.lbl_inference_textual_info.configure(
            width=( 20 + max( [len(line_text) for line_text in self.objects_found_info_arr] )  ),
            height=(2 + len(self.objects_found_info_arr) )
            )
        
        ## pack it all
        self.lbl_inference_img.pack(padx=15, pady=15)
        self.lbl_inference_textual_info.pack(padx=15, pady=15)
        self.lbl_path_img_inferred.pack(padx=15, pady=15)

        ## display the Inference window
        self.wnd_inference_img.mainloop()
        return

class c_keras_inference_performer: ## adapted from jbrownlee/keras-yolo3
    
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
    
    def draw_boxes(self, image, boxes, labels, obj_thresh, _objects_found_info_arr):
        serial_num_object_box_overall = 1
        for box in boxes:
            label_str = ''
            label = -1
            serial_num_object_box_at_loop_start = serial_num_object_box_overall
            for i in range(len(labels)):
                if box.classes[i] > obj_thresh:
                    label_str += labels[i]
                    label = i

                    #print(labels[i] + ': ' + str(box.classes[i]*100) + '%')
                    print(f"{serial_num_object_box_overall}) {labels[i]} : {box.classes[i]*100:.2f}%")
                    _objects_found_info_arr.append( f"{serial_num_object_box_overall}) {labels[i]} :\t\t{box.classes[i]*100:.2f} %" )
                    serial_num_object_box_overall += 1
            
            if label >= 0:
                cv2.rectangle(image, (box.xmin,box.ymin), (box.xmax,box.ymax), (0,255,0), 3)
                # cv2.putText(image, 
                #             label_str + ' ' + str(box.get_score()), 
                #             (box.xmin, box.ymin - 13), 
                #             cv2.FONT_HERSHEY_SIMPLEX, 
                #             1e-3 * image.shape[0], 
                #             (0,255,0), 2)
                text_to_superimpose_in_image = f"{serial_num_object_box_at_loop_start}) {label_str}"
                cv2.putText(image, 
                            text_to_superimpose_in_image, 
                            (box.xmin, box.ymin - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1e-3 * image.shape[0], 
                            (0,255,0), 2)
                serial_num_object_box_at_loop_start += 1
        return image
    
    def perform_model_inference(self, _path_infer_this_image, _objects_found_info_arr):
        print(f"\n\nExecuting model inference on image_to_infer : {_path_infer_this_image}\n")
        
        ## load the keras pretained model if not already loaded
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
        self.draw_boxes(image_to_infer_cv2, boxes, labels, obj_thresh, _objects_found_info_arr)

        ## save the image as intermediate file -- see later whether to return and processing is possible
        cv2.imwrite(r'./intermediate_file_inferenece_image.jpg', (image_to_infer_cv2).astype('uint8') )

def select_images_functionality_for_one_query_result(_DEBUG_SWITCH, _query_num, _each_query_result, _index_positions_to_remove_this_query, _num_candidate_images_before_selection_began):
    
    ## make a root window and show it
    o_queryImgSelection_root_window = c_queryImgSelection_root_window(_DEBUG_SWITCH, _query_num, _each_query_result, _index_positions_to_remove_this_query, _num_candidate_images_before_selection_began)
    o_queryImgSelection_root_window.root.mainloop()

def selection_processing_functionality(_DEBUG_SWITCH):
    ## test data for unit test - actually will be passed from earlier stage (id keyword elements)
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
    
    ## execute gui logic to allow deselection of images, check the number of selections are valid,
    ##         capture the deselected positions
    index_positions_to_remove_all_queries = []
    for query_num, each_query_result in enumerate(database_query_results):
        num_candidate_images_before_selection_began = len(each_query_result)
        index_positions_to_remove_this_query = []
        if _DEBUG_SWITCH:
            print(f"\n\nStarting Selection process for Query {query_num + 1}")
            print(f"Number of images before selection began = {num_candidate_images_before_selection_began}\n")
        select_images_functionality_for_one_query_result(_DEBUG_SWITCH, query_num + 1, each_query_result, index_positions_to_remove_this_query, num_candidate_images_before_selection_began)
        index_positions_to_remove_all_queries.append(index_positions_to_remove_this_query)
        
        print(f"\nCompleted selection process - Query number {query_num + 1}\n")
        if _DEBUG_SWITCH:
            num_images_to_remove = len(index_positions_to_remove_this_query)
            print(f"\nNumber of images Deselected by user = {num_images_to_remove}.\nNumber of images that will remain = { num_candidate_images_before_selection_began - num_images_to_remove }")
    
    ## remove the Deselected images
    final_remaining_images_selected_info = []
    for each_query_result, each_index_positions_to_remove in \
        zip(database_query_results, index_positions_to_remove_all_queries):
        temp_array = [each_image_info for idx, each_image_info in enumerate(each_query_result) if idx not in each_index_positions_to_remove]
        final_remaining_images_selected_info.append(temp_array)
    
    ## show summary info
    if _DEBUG_SWITCH:
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

if __name__ == '__main__':
    DEBUG_SWITCH = True
    selection_processing_functionality(DEBUG_SWITCH)
    print(f"\n\nNormal exit.\n\n")
    