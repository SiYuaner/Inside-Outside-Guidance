#-------------------------------------------------------------------------------
# Name:        Object bounding box label tool
# Purpose:     Label object bboxes for ImageNet Detection data
# Author:      Qiushi
# Created:     06/06/2014

#
#-------------------------------------------------------------------------------
from __future__ import division
from tkinter import *
import tkinter.messagebox as tkMessageBox
from PIL import Image, ImageTk
import os
import sys
import glob
import random
import time
import IOG_loop as iog_loop
import IOG as iog
import cv2
import numpy as np
import torch

MAIN_COLORS = ['darkolivegreen', 'darkseagreen', 'darkorange', 'darkslategrey', 'darkturquoise', 'darkgreen', 'darkviolet', 'darkgray', 'darkmagenta', 'darkblue', 'darkkhaki','darkcyan', 'darkred',  'darksalmon', 'darkslategray', 'darkgoldenrod', 'darkgrey', 'darkslateblue', 'darkorchid','skyblue','yellow','orange','red','pink','violet','green','brown','gold','Olive','Maroon', 'blue', 'cyan', 'black','olivedrab', 'lightcyan', 'silver']
color_list = [[0,0,128]  ,[0,128,0] ,[0,128,128] ,[128,0,0] ,[128,0,128] ,[128,128,128] ,[128,128,192]  ]
              #[0,128,0],[128,128,0],[0,0,128] ,[128,0,128],[128,128,128],[192,128,128]   ]
# image sizes for the examples
SIZE = 256, 256

classes = []

try:
    with open('classes.txt','r') as cls:
        classes = cls.readlines()
    classes = [cls.strip() for cls in classes]
except IOError as io:
    print("[ERROR] Please create classes.txt and put your all classes")
    sys.exit(1)
COLORS = random.sample(set(MAIN_COLORS), len(classes))

class LabelTool():
    def __init__(self, master):
        #time 
        self.time_start_index = 0
        self.time_start= 0
        self.time_start_single= 0
        self.time_end= 0
        self.time_end_box= 0
        self.box_num = 0
        self.total_time_box= 0
        
        # set up the main frame
        self.curimg_h = 0
        self.curimg_w = 0
        self.cur_cls_id = -1
        self.parent = master
        self.parent.title("IOG Annotation Tool")
        self.frame = Frame(self.parent)
        self.frame.pack(fill=BOTH, expand=1)
        self.parent.resizable(width = FALSE, height = FALSE)

        # initialize global state
        self.imageDir = ''
        self.imageList= []
        self.egDir = ''
        self.egList = []
        self.outDir = ''
        self.cur = 0
        self.total = 0
        self.category = 0
        self.imagename = ''
        self.labelfilename = ''
        self.tkimg = None

        # initialize mouse state
        self.STATE = {}
        self.STATE['click'] = 0
        self.STATE['x'], self.STATE['y'] = 0, 0
        self.STATE['x3'], self.STATE['y3'] = -100, -100

        # reference to bbox
        self.bboxIdList = []
        self.bboxId = None
        self.bboxList = []
        self.bboxListCls = []
        self.hl = None
        self.vl = None

        # ----------------- GUI stuff ---------------------
        # dir entry & load
        self.label = Label(self.frame,text = "Image Dir:")
        self.label.grid(row = 0, column = 0, sticky = E)
        
        
        self.entry = Entry(self.frame)
        self.entry.focus_set()
        self.entry.bind('<Return>', self.loadEntry)
        self.entry.grid(row = 0, column = 1, sticky = W+E)
#        self.ldBtn = Button(self.frame, text = "Load", command = self.loadDir)
#        self.ldBtn.grid(row = 0, column = 2, sticky = W+E)

        # add iog
        self.netindex = None        
        self.IOGPanel = Frame(self.frame)
        self.IOGPanel.grid(row = 5, column = 1, columnspan = 2, sticky = W+E)                
#        self.IOG_Btn = Button(self.frame, text = "IOG_initial", command = self.noloop_initial)
#        self.IOG_Btn.grid(row = 6, column = 1, sticky = W+E)
#        self.IOG_Btn = Button(self.IOGPanel, text = "IOG initial", command = self.noloop_initial)
#        self.IOG_Btn.pack(side = LEFT,  expand= YES,fill=X )        
        self.IOG_bgpoint = [0,0,0,0]
        self.IOG_cppoint = [0,0]
        self.IOG_image = None
        self.IOG_net = None
        self.imgwith_mask = None
        self.imgwith_mask_lastobject = None
        
        # add iog inloop
#        self.IOGloop_Btn = Button(self.frame, text = "IOG_loop_initial", command = self.loop_initial)
#        self.IOGloop_Btn.grid(row = 6, column = 2, sticky = W+E)
#        self.IOGloop_Btn = Button(self.IOGPanel, text = "Interactivate Refinement", command = self.loop_initial)
#        self.IOGloop_Btn.pack(side = LEFT, expand= YES,fill=X) 
        self.IOGloop_feature = None
        self.newpoint_list = 0
        self.color = (0,0,128)
        self.nextcolor = 0
#        self.bgIOGloop_Btn = Button(self.frame, text = "on the bg", command = self.bg_box)
#        self.bgIOGloop_Btn.grid(row = 7, column = 2, sticky = W+E)  
#        self.cpIOGloop_Btn = Button(self.frame, text = "on the cp", command = self.cp_box)
#        self.cpIOGloop_Btn.grid(row = 8, column = 2, sticky = W+E)  
        self.switch = -1
        # next box
#        self.IOGloop_Btn = Button(self.frame, text = "next object", command = self.next_box)
#        self.IOGloop_Btn.grid(row = 9, column = 2, sticky = W+E)   
  
        self.prevBtn = Button(self.IOGPanel,bg = '#526069',fg = '#ffffff', text='<< Prev', width = 10, command = self.prevImage)
        self.prevBtn.pack(side = LEFT, expand= YES,fill=X)
        self.nextBtn = Button(self.IOGPanel,bg = '#526069',fg = '#ffffff', text='Next >>', width = 10, command = self.nextImage)
        self.nextBtn.pack(side = LEFT,expand= YES,fill=X)        
        self.IOGloop_Btn = Button(self.IOGPanel,bg = '#526069',fg = '#ffffff', text = "Next object", command = self.next_box)
        self.IOGloop_Btn.pack(side = LEFT,  expand= YES,fill=X)          
        
        # main panel for labeling
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        
#        self.mainPanel.bind("<Double-Button-3>", self.bg_box) ####zsy
        self.mainPanel.bind("<Button-3>", self.bg_box) ####zsy
        
        self.mainPanel.bind("<Button-1>", self.mouseClick)
        self.mainPanel.bind("<Motion>", self.mouseMove)
        self.parent.bind("<Escape>", self.cancelBBox)  # press <Espace> to cancel current bbox
        self.parent.bind("s", self.cancelBBox)
        self.parent.bind("<Left>", self.prevImage) # press 'a' to go backforward
        self.parent.bind("<Right>", self.nextImage) # press 'd' to go forward
        self.mainPanel.grid(row = 1, column = 1, rowspan = 4, sticky = W+N)

        # showing bbox info & delete bbox
#        self.tkvar = StringVar(self.parent)
        self.cur_cls_id = 0
#        self.tkvar.set(classes[0]) # set the default option
#        self.popupMenu = OptionMenu(self.frame, self.tkvar, *classes,command = self.change_dropdown)
#        self.popupMenu.grid(row = 1, column =2, sticky = E+S)
#        self.chooselbl = Label(self.frame, text = 'Choose Class:')
#        self.chooselbl.grid(row = 1, column = 2, sticky = W+S)
#        self.lb1 = Label(self.frame, text = 'Bounding boxes & points:')
#        self.lb1.grid(row = 2, column = 2,  sticky = W+N)
#        self.listbox = Listbox(self.frame, width = 30+10, height = 12)
#        self.listbox.grid(row = 3, column = 2, sticky = N)
#        self.btnDel = Button(self.frame, text = 'Delete', command = self.delBBox)
#        self.btnDel.grid(row = 4, column = 2, sticky = W+E+N)
#        self.btnClear = Button(self.frame, text = 'Clear', command = self.clearBBox)
#        self.btnClear.grid(row = 10, column = 2, sticky = W+E+N) #W+E+N)
        self.btnClear = Button(self.IOGPanel,bg = '#526069',fg = '#ffffff', text = 'Clear', command = self.clearBBox)
        self.btnClear.pack(side = LEFT,  expand= YES,fill=X)  

        # control panel for image navigation
        self.ctrPanel = Frame(self.frame)
        self.ctrPanel.grid(row = 11, column = 1, columnspan = 2, sticky = W+E)
        self.IOG_Btn = Button(self.ctrPanel,bg = '#526069',fg = '#ffffff', text = "IOG initialization", command = self.noloop_initial)
        self.IOG_Btn.pack(side = LEFT,  expand= YES,fill=X )  
        self.IOGloop_Btn = Button(self.ctrPanel,bg = '#526069',fg = '#ffffff', text = "Interactivate Refinement", command = self.loop_initial)
        self.IOGloop_Btn.pack(side = LEFT, expand= YES,fill=X)         
#        self.prevBtn = Button(self.ctrPanel, text='<< Prev', width = 10, command = self.prevImage)
#        self.prevBtn.pack(side = LEFT, expand= YES,fill=X)
#        self.nextBtn = Button(self.ctrPanel, text='Next >>', width = 10, command = self.nextImage)
#        self.nextBtn.pack(side = LEFT,expand= YES,fill=X)
#        self.progLabel = Label(self.ctrPanel, text = "Progress:     /    ")
#        self.progLabel.pack(side = LEFT, padx = 5)
#        self.tmpLabel = Label(self.ctrPanel, text = "Go to Image No.")
#        self.tmpLabel.pack(side = LEFT, padx = 5)
#        self.idxEntry = Entry(self.ctrPanel, width = 5)
#        self.idxEntry.pack(side = LEFT)
#        self.goBtn = Button(self.ctrPanel, text = 'Go', command = self.gotoImage)
#        self.goBtn.pack(side = LEFT)

        # example pannel for illustration
        self.egPanel = Frame(self.frame, border = 10)
        self.egPanel.grid(row = 1, column = 0, rowspan = 5, sticky = N)
        self.tmpLabel2 = Label(self.egPanel, text = "Examples:")
        self.tmpLabel2.pack(side = TOP, pady = 5)
        self.egLabels = []
        for i in range(3):
            self.egLabels.append(Label(self.egPanel))
            self.egLabels[-1].pack(side = TOP)

        # display mouse position
        self.disp = Label(self.ctrPanel, text='')
        self.disp.pack(side = RIGHT)

        self.frame.columnconfigure(1, weight = 1)
        self.frame.rowconfigure(4, weight = 1)
        
    def loop_initial(self):
        self.IOG_net = iog_loop.loadnetwork()
        self.netindex = 1
    def noloop_initial(self):
        self.IOG_net = iog.loadnetwork()
        self.netindex = 0
        
    def next_box(self):
        self.STATE['click'] = 0
        self.newpoint_list = 0
        self.IOGloop_feature = None
        self.switch = -1
        self.imgwith_mask_lastobject = self.imgwith_mask
        
    def bg_box(self,event):
        if self.STATE['click']>=3:
            print('bg box')
            self.switch = 1
            self.STATE['x_add'], self.STATE['y_add'] = event.x, event.y
            self.STATE['click'] = 4
            self.newpoint_list=1        
            self.IOG_loop()
        
    def cp_box(self):
        if self.netindex ==1:
            self.switch = 0
            self.STATE['click'] = 4
            self.newpoint_list=1          
            self.IOG_loop()
            print('cp box')
#    def cp_box(self,event):
#        self.switch = 0
#        self.IOG_loop()
#        print('cp box')
        
    def getwhitemask(self,cp,shape,index,color=(255,255,255),edge_w = 5):

        cp = cp.astype(np.uint8)
        ret,binaary = cv2.threshold(cp,1,255,cv2.THRESH_BINARY)
        contour,he = cv2.findContours(binaary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cp_edge = np.zeros(shape)
        if index == 'edge':
            cv2.drawContours(cp_edge,contour,-1,color,edge_w)   
        if index == 'mask':
            cv2.drawContours(cp_edge,contour,-1,color,-1) 
        
        return cp_edge
        
    def mixmask(self,outputs_image,cp_mask,hard=1):
    
        cp_mask = np.uint8(cp_mask)
        if len(cp_mask.shape) == 3:
            mask = cv2.cvtColor(cp_mask,cv2.COLOR_BGR2GRAY)
        else:
            mask = cp_mask
        ret,mask_th = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask_th)
        img_bg = cv2.bitwise_and(outputs_image,outputs_image,mask =mask_inv )
        
        if hard ==1:
            mixx = cv2.addWeighted(outputs_image,0.2,cp_mask,0.8,0)
            img_ob = cv2.bitwise_and(mixx,mixx,mask =mask_th)
            outimg = cv2.add(img_ob,img_bg)        
#            img_ob = cv2.bitwise_and(cp_mask,cp_mask,mask =mask_th)
#            outimg = cv2.add(img_ob,img_bg)
        else:
            mixx = cv2.addWeighted(outputs_image,0.6,cp_mask,0.4,0)
            img_ob = cv2.bitwise_and(mixx,mixx,mask =mask_th)
            outimg = cv2.add(img_ob,img_bg)
            
        return outimg        
        
    def IOG(self):
        print('start get mask')
#        print(self.IOG_bgpoint,self.IOG_cppoint)
        mask = iog.IOG_getmask(self.IOG_bgpoint,self.IOG_cppoint,self.IOG_image,self.IOG_net)
        img = self.imgwith_mask#self.IOG_image
        result_mask = self.getwhitemask(mask,img.shape,'mask',self.color)                 
        final_mask = self.mixmask(img,result_mask,hard =0)   
        result_edge = self.getwhitemask(mask,img.shape,'edge',(10,10,10),edge_w = 1)
        final_mask = self.mixmask(final_mask,result_edge,hard =1)   
        
        self.imgwith_mask = final_mask
        final_mask_rgb = np.zeros(final_mask.shape)
        final_mask_rgb[:,:,0] = final_mask[:,:,2]
        final_mask_rgb[:,:,1] = final_mask[:,:,1]
        final_mask_rgb[:,:,2] = final_mask[:,:,0]
        img_rgb = Image.fromarray(np.uint8(final_mask_rgb))
        
        self.tkimg = ImageTk.PhotoImage(img_rgb)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)  
        self.next_box()
        
    def IOG_loop(self):
        print('start get mask in loop')   
        if self.newpoint_list == 0:
            loopsignal = -1
            img = self.imgwith_mask#
        else:
            loopsignal = [self.STATE['x_add'], self.STATE['y_add'] ]
            img = self.imgwith_mask_lastobject #self.imgwith_mask#
            
        mask,self.IOGloop_feature = iog_loop.IOG_getmask(self.IOG_bgpoint,self.IOG_cppoint,self.IOG_image,self.IOG_net,\
                                                         loopsignal,self.switch,self.IOGloop_feature)
        #switch = True #if the new point is on the object: true, if the new point is on the bg: false
        


#        img = self.IOG_image #self.imgwith_mask#
        result_mask = self.getwhitemask(mask,img.shape,'mask',self.color) # (0,0,128)               
        final_mask = self.mixmask(img,result_mask,hard =0)  
        result_edge = self.getwhitemask(mask,img.shape,'edge',(10,10,10),edge_w = 1)
        final_mask = self.mixmask(final_mask,result_edge,hard =1)           
        
        self.imgwith_mask = final_mask
        final_mask_rgb = np.zeros(final_mask.shape)
        final_mask_rgb[:,:,0] = final_mask[:,:,2]
        final_mask_rgb[:,:,1] = final_mask[:,:,1]
        final_mask_rgb[:,:,2] = final_mask[:,:,0]
        img_rgb = Image.fromarray(np.uint8(final_mask_rgb))
        
        self.tkimg = ImageTk.PhotoImage(img_rgb)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)   
        
        
    def loadEntry(self,event):
        self.loadDir()

    def loadDir(self, dbg = False):
        if not dbg:
            try:
                s = self.entry.get()
                self.parent.focus()
                self.category = s
            except ValueError as ve:
                tkMessageBox.showerror("Error!", message = "The folder should be numbers")
                return
        if not os.path.isdir('./Images/%s' % self.category):
           print(os.path.isdir('./Images/%s' % self.category))
           tkMessageBox.showerror("Error!", message = "The specified dir doesn't existzsy!")
           return
        # get image list
        self.imageDir = os.path.join(r'./Images', '%s' %(self.category))
        self.imageList = glob.glob(os.path.join(self.imageDir, '*.jpg'))
        self.imageList.sort()
        if len(self.imageList) == 0:
            print('No .jpg images found in the specified dir!')
            tkMessageBox.showerror("Error!", message = "No .jpg images found in the specified dir!")
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

         # set up output dir
        if not os.path.exists('./Labels'):
            os.mkdir('./Labels')
        self.outDir = os.path.join(r'./Labels', '%s' %(self.category))
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)
        self.loadImage()
        print ('%d images loaded from %s' %(self.total, s))

    def loadImage(self):
        # load image
        print('load img')
        self.imgwith_mask_lastobject = None
        self.nextcolor = 0
        imagepath = self.imageList[self.cur - 1]
        self.IOG_image = cv2.imread(imagepath)
        self.imgwith_mask = cv2.imread(imagepath)
        self.img = Image.open(imagepath)
        
        w,h,c = self.IOG_image.shape
        if 1:#h>500:
            print('resize')
            h = int(  h*700/w)
#            w=500
            w = 700
            self.IOG_image = cv2.resize(self.IOG_image,(h,w))
            self.imgwith_mask = cv2.resize(self.imgwith_mask,(h,w))
            self.img = self.img.resize((h,w))
            print(self.img.size)
        self.curimg_w, self.curimg_h = self.img.size
        self.tkimg = ImageTk.PhotoImage(self.img)
        self.mainPanel.config(width = max(self.tkimg.width(), 400), height = max(self.tkimg.height(), 400))
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
#        self.progLabel.config(text = "%04d/%04d" %(self.cur, self.total))

        # load labels
        self.clearBBox()
        # self.imagename = os.path.split(imagepath)[-1].split('.')[0]
        self.imagename = os.path.splitext(os.path.basename(imagepath))[0]
        labelname = self.imagename + '.txt'
        self.labelfilename = os.path.join(self.outDir, labelname)
        bbox_cnt = 0
        if os.path.exists(self.labelfilename):
            with open(self.labelfilename) as f:
                for (i, line) in enumerate(f):
                    yolo_data = line.strip().split()
                    #print(yolo_data)
                    tmp = self.deconvert(yolo_data[1:])
                    self.bboxList.append(tuple(tmp))
                    self.bboxListCls.append(yolo_data[0])
                    tmpId = self.mainPanel.create_rectangle(tmp[0], tmp[1], \
                                                            tmp[2], tmp[3], \
                                                            width = 2, \
                                                            outline = COLORS[int(yolo_data[0])])
                    r = 1          
                    tmpId_cp = self.mainPanel.create_oval(tmp[4]-r, tmp[5]-r, tmp[4]+r, tmp[5]+r, width=5,\
                                                          outline = COLORS[int(yolo_data[0])] )                                       
                    
                    self.bboxIdList.append(tmpId)
#                    self.listbox.insert(END, '(%d, %d) -> (%d, %d) -> (%d, %d)' %(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5] ))
#                    self.listbox.insert(END, '(%d, %d) -> (%d, %d) -> (%s)' %(tmp[0], tmp[1], tmp[2], tmp[3], classes[int(yolo_data[0])]))
#                    self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[int(yolo_data[0])])

        #start time
        if self.time_start_index ==0:
            self.time_start = time.time()
            self.time_start_single = time.time()
            self.time_start_index = 1
            print('start time',self.time_start )     
               
    def saveImage(self):
#        with open(self.labelfilename, 'w') as f:
#            for bbox,bboxcls in zip(self.bboxList,self.bboxListCls):
#                xmin,ymin,xmax,ymax ,cx,cy= bbox
#                b = (float(xmin), float(xmax), float(ymin), float(ymax), float(cx), float(cy))
#                bb = self.convert((self.curimg_w,self.curimg_h), b)
#                f.write(str(bboxcls) + " " + " ".join([str(a) for a in bb]) + '\n')
        print ('Image No. %d saved' %(self.cur))

    def mouseClick(self, event):

        if self.STATE['click'] == 0:
            self.next_box()#####clear
            self.color = color_list[self.nextcolor]
            self.nextcolor = self.nextcolor+1
            self.STATE['x'], self.STATE['y'] = event.x, event.y
            self.IOG_bgpoint[0],self.IOG_bgpoint[1] = event.x, event.y
            self.STATE['click'] = 1
        elif self.STATE['click'] == 1:
            self.STATE['x2'], self.STATE['y2'] = event.x, event.y
            self.IOG_bgpoint[2],self.IOG_bgpoint[3] = event.x, event.y
            self.STATE['click'] = 2
            self.time_end_box = time.time()
            self.total_time_box = self.total_time_box +(self.time_end_box-self.time_start_single)
        elif self.STATE['click'] == 2:
            self.STATE['x3'], self.STATE['y3'] = event.x, event.y
            self.IOG_cppoint[0],self.IOG_cppoint[1] = event.x, event.y
            x1, x2 = min(self.STATE['x'], self.STATE['x2']), max(self.STATE['x'], self.STATE['x2'])
            y1, y2 = min(self.STATE['y'], self.STATE['y2']), max(self.STATE['y'], self.STATE['y2'])    
            cx,cy = self.STATE['x3'], self.STATE['y3'] 
            self.bboxList.append((x1, y1, x2, y2,cx,cy))
            self.bboxListCls.append(self.cur_cls_id)
            self.bboxIdList.append(self.bboxId)
            self.bboxId = None      
#            self.listbox.insert(END, '(%d, %d) -> (%d, %d) -> (%d, %d)' %(x1, y1, x2, y2,cx,cy))
#            self.listbox.itemconfig(len(self.bboxIdList) - 1, fg = COLORS[self.cur_cls_id])                                      
            self.STATE['click'] = 3

            #time end
            self.time_end_single = time.time()
            self.time_end = time.time()
            self.box_num = self.box_num+1
            self.time_start_single = time.time()
            if self.switch == -1:
                if self.netindex ==0:
                    self.IOG() 
                else:
                    self.IOG_loop()            
        else:
            self.STATE['x_add'], self.STATE['y_add'] = event.x, event.y
            self.cp_box()
            self.STATE['click'] = 4
            self.newpoint_list=1


    def mouseMove(self, event):
        self.disp.config(text = 'x: %d, y: %d' %(event.x, event.y))
        if self.tkimg:
            if self.hl:
                self.mainPanel.delete(self.hl)
            self.hl = self.mainPanel.create_line(0, event.y, self.tkimg.width(), event.y, width = 2, fill= 'white')
            if self.vl:
                self.mainPanel.delete(self.vl)
            self.vl = self.mainPanel.create_line(event.x, 0, event.x, self.tkimg.height(), width = 2, fill ='white' )
            
        if 1 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            self.bboxId = self.mainPanel.create_rectangle(self.STATE['x'], self.STATE['y'], \
                                                            event.x, event.y, \
                                                            width = 2, \
                                                            outline = 'green')#COLORS[self.cur_cls_id])
          
        if 3 == self.STATE['click']:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            x = self.STATE['x3']
            y = self.STATE['y3']
            r = 1
            x0 = x - r
            y0 = y - r
            x1 = x + r
            y1 = y + r            
            self.bboxId = self.mainPanel.create_oval(x0, y0, x1, y1, width=5,outline = 'orange')#COLORS[self.cur_cls_id])

#            self.STATE['click'] = 0 ##############straightly go to another box
        if self.STATE['click']>3:
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
            x = self.STATE['x_add']
            y = self.STATE['y_add']
            r = 1
            x0 = x - r
            y0 = y - r
            x1 = x + r
            y1 = y + r            
            self.bboxId = self.mainPanel.create_oval(x0, y0, x1, y1, width=5,outline = COLORS[self.cur_cls_id])

    def cancelBBox(self, event):
        if 3 == self.STATE['click']:
            print('cancelBBox')
            if self.bboxId:
                self.mainPanel.delete(self.bboxId)
                self.bboxId = None
                self.STATE['click'] = 0
        else:
            print('no cancelBBox')



    def delBBox(self):
        
#        sel = self.listbox.curselection()
        if len(sel) != 1 :
            return
        idx = int(sel[0])
        self.mainPanel.delete(self.bboxIdList[idx])
        self.bboxIdList.pop(idx)
        self.bboxList.pop(idx)
        
        self.bboxListCls.pop(idx)
#        self.listbox.delete(idx)

     
        
    def clearBBox(self):
        self.STATE['click'] == 0
        for idx in range(len(self.bboxIdList)):
            self.mainPanel.delete(self.bboxIdList[idx])
#        self.listbox.delete(0, len(self.bboxList))
        self.bboxIdList = []
        self.bboxList = []
        self.bboxListCls = []

        
    def prevImage(self, event = None):
        self.saveImage()
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()
        else:
            tkMessageBox.showerror("Information!", message = "This is first image")

    def nextImage(self, event = None):
        self.STATE['click'] = 0#####
        self.saveImage()
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()
        else:
            tkMessageBox.showerror("Information!", message = "All images annotated")

    def gotoImage(self):
        idx = int(self.idxEntry.get())
        if 1 <= idx and idx <= self.total:
            self.saveImage()
            self.cur = idx
            self.loadImage()
    def change_dropdown(self,*args):
        cur_cls = self.tkvar.get()
        self.cur_cls_id = classes.index(cur_cls)
        
    def convert(self,size, box):
        dw = 1./size[0]
        dh = 1./size[1]
        x = (box[0] + box[1])/2.0
        y = (box[2] + box[3])/2.0
        w = box[1] - box[0]
        h = box[3] - box[2]
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        cx = box[4]*dw
        cy = box[5]*dh
        return (x,y,w,h,cx,cy)

    def deconvert(self,annbox):
        ox = float(annbox[0])
        oy = float(annbox[1])
        ow = float(annbox[2])
        oh = float(annbox[3])
        ocx = float(annbox[4])
        ocy = float(annbox[5])
        x = ox*self.curimg_w
        y = oy*self.curimg_h
        w = ow*self.curimg_w
        h = oh*self.curimg_h
        xmax = (((2*x)+w)/2)
        xmin = xmax-w
        ymax = (((2*y)+h)/2)
        ymin = ymax-h
        
        cx = ocx*self.curimg_w
        cy = ocy*self.curimg_h
        return [int(xmin),int(ymin),int(xmax),int(ymax),int(cx),int(cy)]

if __name__ == '__main__':
    root = Tk()
    root.tk_setPalette(background = '#19282e')
    tool = LabelTool(root)
    root.resizable(width =  True, height = True)
    root.mainloop()
