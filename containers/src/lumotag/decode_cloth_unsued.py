
def analyse_candidates(
        original_img,
        original_img_grayscale,
        contours : tuple [np.ndarray],
        dataobject : WorkingData):
    """supply monitoring image and image which has masked area of the irregular contours found
    for ID patches (not rectangular bounding boxes
    
    original_image = np array n/n/3 (colour image)
    masked_img = binary image
    contour"""
    playerfound = [False]
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        original_samp = original_img_grayscale[y:y + h, x:x + w].copy()
        original_samp=img_pro.normalise(original_samp)
        original_samp=img_pro.threshold_img_static(original_samp, low = 127, high = 255)
        decoded_ID, good_result= decode_ID_image(original_samp, dataobject)
        playerfound.append(good_result)
        if decoded_ID is not None:#
            #original_img[y:y + h, x:x + w,0:3] = decoded_ID#cv2.cvtColor(decoded_ID,cv2.COLOR_BGR2GRAY)
            original_img = cv2.rectangle(
                    original_img,
                    (x, y),
                    (x + w, y + h),
                    (0,0, 255),
                    5
                    )
            dataobject.img_view_or_save_if_debug(original_samp, Debug_Images.ID_BADGE.value)


    return original_img, any(playerfound)

def decode_ID_image(img,dataobject : WorkingData):
    """provide single ID, thresholded binary image
    cv2.RETR_TREE will return a hierachy where we can find a parent
    contour with N children. 
    The following two outputs are from valid IDs with 4 elements in 
    a block. Each internal element has no children so the 3rd
    value should be -1, and each will have the same parent. Thereby
    the rule may be that if there are 4 instances in the 4th column of
    the parent contour, and each child has no other children (-1 in 3rd
    column), then we can mark this ID as a candidate
    cv2.findContours hierachy output
    ex 1
    [[[ 5 -1  1 -1]
    [ 2 -1 -1  0]
    [ 3  1 -1  0]
    [ 4  2 -1  0]
    [-1  3 -1  0]
    [-1  0 -1 -1]]]
    ex 2
    [[[ 1 -1 -1 -1]
    [ 2  0 -1 -1]
    [ 3  1 -1 -1]
    [-1  2  4 -1]
    [ 5 -1 -1  3]
    [ 6  4 -1  3]
    [ 7  5 -1  3]
    [-1  6 -1  3]]]
    
    index = contourid

    |next cnt in same tier|previous cnt in same tier|child|parent|"""
    
    # create ID badge
    id_badge = np.zeros((50,50,3), np.uint8)
    id_badge[:,:,1] = 255
    id_badge = cv2.putText(id_badge, "P3", (4,40), cv2.FONT_HERSHEY_SIMPLEX,fontScale=1,color=(0,0,0),thickness=2)
    id_badge = cv2.rotate(id_badge, cv2.ROTATE_90_CLOCKWISE)
    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    contours, hierarchy = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)#RETR_EXTERNAL #RETR_TREE
     
    if contours is None or len(contours) == 0:
        
        #dataobject.img_view_or_save_if_debug(img,f"{Debug_Images.ERROR_no_contours.value}")
        return None, False
    #img_check_contours = img.copy()
    #img_check_contours = cv2.cvtColor(img_check_contours,cv2.COLOR_GRAY2BGR)
    #cv2.drawContours(image=img_check_contours, contours=contours, contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    contours_area = []


    #img_check_contours = img.copy()
    #img_check_contours = np.zeros_like(cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB))
    
    #for i, cnt in enumerate(contours):
    #    cv2.drawContours(image=img_check_contours, contours=[cnt], contourIdx=-1, color=(255,int(255/(i+1)),int(255/(i+1))), thickness=1, lineType=cv2.LINE_AA)

    #np.diff(np.where(has_children == -1))
    #print("checking check id input")
    #_3DVisLabLib.ImageViewer_Quick_no_resize(img,0,True,False)
    #print("cnts before anything filters", len(contours))
    # calculate area and filter into new array
    for con, hier in zip(contours,hierarchy[0]):
        area = cv2.contourArea(con)
        if 25 < area < 1000000:
            contours_area.append((con, hier))
    
    contours_cirles = []
    #print("cnts after area filter", len(contours_area))
    # check if contour is of circular shape
    circularities = []
    for con, hier in contours_area:
        perimeter = cv2.arcLength(con, True)
        area = cv2.contourArea(con)
        if perimeter == 0:
            break
        circularity = 4*math.pi*(area/(perimeter*perimeter))
        if -9999 < circularity < 9999:
            contours_cirles.append((con, hier))
            circularities.append(circularity)
    #print("cnts after circle filter", len(contours_cirles))

    filtered_hierarchy = np.expand_dims(np.array([i[1] for i in contours_cirles]), axis=0)
    img_check_contours = None #LAZY CODE do this properly 
    if dataobject.debug is True:
        img_check_contours = img.copy()
        #img_check_contours = cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB)
        img_check_contours = np.zeros_like(cv2.cvtColor(img_check_contours, cv2.COLOR_GRAY2RGB))
        
        #cv2.drawContours(image=img_check_contours, contours=[i[0] for i in contours_cirles], contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        
        for i, cnt in enumerate([i[0] for i in contours_cirles]):
            cv2.drawContours(image=img_check_contours, contours=[cnt], contourIdx=-1, color=(random.randint(50,255),random.randint(50,255),int(255/(i+1))), thickness=1, lineType=cv2.LINE_AA)
            #_3DVisLabLib.ImageViewer_Quick_no_resize(img_check_contours,0,True,False)
    #dataobject.img_view_or_save_if_debug(img_check_contours,f"Debug_Images.fitered_contours.value{len(contours_cirles)}")

    if len(contours_cirles) > 4 : # ID body and 4 internal markers:
        pass
        #dataobject.img_view_or_save_if_debug(img_check_contours,f"Debug_Images.GOOD_CANDIDATE_ContourCount.value{len(contours_cirles)}")
    else:
        return None, False

    if not check_ID_contours_match_spec(filtered_hierarchy, circularities):
        #_3DVisLabLib.ImageViewer_Quick_no_resize(img_check_contours,0,True,False)
        return None, False
    #_3DVisLabLib.ImageViewer_Quick_no_resize(img_check_contours,0,True,False)
    dataobject.img_view_or_save_if_debug(img_check_contours,f"POSITIVE_ID_ContourCount{len(contours_cirles)}")


    out = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    id_badge = cv2.resize(id_badge,(out.shape[1],out.shape[0]))
    #print(len(contours_cirles))

    cv2.drawContours(image=out, contours=[i[0] for i in contours_cirles], contourIdx=-1, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    #_3DVisLabLib.ImageViewer_Quick_no_resize(out,0,True,False)
    #cv2.drawContours(out, contours_cirles, , 255,1)
    return id_badge, True

def check_ID_contours_match_spec(hierarchy, circularities):
    #print(hierarchy)
    temp_hier=[]
    min_circularity = 0.75
    min_no_circles = 1
    for _index  in range (len(hierarchy[0])):
        if hierarchy[0][_index][2] == -1: # cont has no children
            if hierarchy[0][_index][3] != -1: # cont parent is not top level
                temp_hier.append(hierarchy[0][_index])
    for _index  in range (len(hierarchy[0])):
        if hierarchy[0][_index][2] == -1: #contour with no children
            # check if we have a group of 4 contours with
            # no children
            # get current position and following 3
            child_block = hierarchy[0][_index:_index+4,2]
            parent_block = hierarchy[0][_index:_index+4,3]
            # NB "set" removes duplicates
            if all([len(set(child_block)) == 1,
                    len(child_block) == 4,
                    len(set(parent_block)) == 1,
                    len([i for i in circularities[_index:_index+4] if i > min_circularity])>min_no_circles]):
                return True
            else:
                # jump to next block rather than reprocess
                _index = _index + 4
                continue
    return False


def get_tiled_intensity(img, n_tiles_edge):
    """input mono image
    returns list(int), list(int)of max/min values per tile"""
    edge_len_pxls = int(img.shape[0]/n_tiles_edge)
    if edge_len_pxls <20:
        # for production use logging and send error, don't crash out
        raise ValueError("length not valid for lumo application")
    toprange=img.shape[0]-(img.shape[0]%edge_len_pxls) # ignore remainder (probably should centre it)
    siderange=img.shape[1]-(img.shape[1]%edge_len_pxls)
    print("----------------------------------")
    maxes=[]
    mins=[]
    for vert in range(0,toprange,edge_len_pxls):
        for horiz in range(0,siderange,edge_len_pxls):
            maxes.append(img[vert:vert+edge_len_pxls,horiz:horiz+edge_len_pxls].max())
            mins.append(img[vert:vert+edge_len_pxls,horiz:horiz+edge_len_pxls].min())

    # dont need to sort them 
    maxes.sort()
    mins.sort()
    return maxes, mins