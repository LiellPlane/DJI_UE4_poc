        #get grayscale
        Img_grayscale=cv2.cvtColor(Img_colour, cv2.COLOR_BGR2GRAY)
        #get HSV (may work better than RGB)
        Img_colour_HSV=cv2.cvtColor(Img_colour,cv2.COLOR_BGR2HSV)
        #_3DVisLabLib.ImageViewer_Quick_no_resize(Img_colour,0,True,True)


        #HISTOGRAM STRIPING - vertical and horizontal histogram 
        if "HM_data_HistogramStriping" in Metrics_dict:
            global HistogramStripulus_Centralis
            if HistogramStripulus_Centralis is None:#check global object is populated with mask or not
               HistogramStripulus_Centralis = np.zeros((Img_colour_HSV.shape[0],Img_colour_HSV.shape[1],3), np.uint8)
               #get radius
               Diameter=min((HistogramStripulus_Centralis.shape[0]),(HistogramStripulus_Centralis.shape[1]))
               Radius=int((Diameter/2)*1)#percentage of smallest dimension (1=100%)
               cv2.circle(HistogramStripulus_Centralis,(int(HistogramStripulus_Centralis.shape[1]/2),int(HistogramStripulus_Centralis.shape[0]/2)), Radius, (255,255,255), -1)
               HistogramStripulus_Centralis = cv2.cvtColor(HistogramStripulus_Centralis, cv2.COLOR_BGR2GRAY)
            #MacroStructure_img = cv2.resize(Img_colour_HSV, (25, 25))
            #_3DVisLabLib.ImageViewer_Quick_no_resize(HistogramStripulus_Centralis,0,True,True)
            HistoStripes = cv2.filter2D(Img_colour_HSV,-1,kernel)
            HistoStripes=Histogram_Stripes(HistoStripes,7,8,HistogramStripulus_Centralis)
            ImageInfo.Metrics_functions["HM_data_HistogramStriping"].append(HistoStripes)

        #MACROSTRUCTURE
        if "HM_data_MacroStructure" in Metrics_dict:
        #get small image to experiment with macro structure matching
            
            MacroStructure_img = cv2.resize(Img_grayscale, (19, 19))
            KernelSize=3
            kernel = np.ones((KernelSize,KernelSize),np.float32)/(KernelSize*KernelSize)#kernel size for smoothing
            MacroStructure_img = cv2.filter2D(MacroStructure_img,-1,kernel)
            #MacroStructure_img=MacroStructure_img/np.sqrt(np.sum(MacroStructure_img**2))
            ImageInfo.Metrics_functions["HM_data_MacroStructure"].append(MacroStructure_img)
            #_3DVisLabLib.ImageViewer_Quick_no_resize(MacroStructure_img,0,True,True)
#

def Histogram_Stripes(Image,StripsPerDim,BinsPerChannel,Mask):
    #generates horizontal and vertical histogram stripes which may be compared to
    #overcome skewed and aspect ratio'd images
    Height=Image.shape[0]
    Width=Image.shape[1]
    Histogram_List=[]

    HeightStrips=floor(Height/StripsPerDim)
    WidthStrips=floor(Width/StripsPerDim)

    for WidthStripeIndex in range (0,Width,WidthStrips):
        #Image_for_slice=Image[:,WidthStripeIndex:WidthStripeIndex+WidthStrips]
        hist = cv2.calcHist([Image], [0, 1, 2], Mask, [BinsPerChannel, BinsPerChannel, BinsPerChannel],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        Histogram_List.append(hist)
    for HeightStripeIndex in range (0,Height,HeightStrips):
        #Image_for_slice=Image[HeightStripeIndex:HeightStripeIndex+HeightStrips,:]
        hist = cv2.calcHist([Image], [0, 1, 2], Mask, [BinsPerChannel, BinsPerChannel, BinsPerChannel],[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        Histogram_List.append(hist)

    return Histogram_List