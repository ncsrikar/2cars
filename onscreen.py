import numpy as np
import cv2
import glob
from moviepy.editor import VideoFileClip
from mss import mss
from PIL import Image
from darkflow.net.build import TFNet
import time

options = {
    'pbLoad' : 'C:\\Users\\Srikar\\Desktop\\carsAI\\darkflow\\built_graph\\tiny-yolo-voc-3c.pb' ,
   'metaLoad': 'C:\\Users\\Srikar\\Desktop\\carsAI\\darkflow\\built_graph\\tiny-yolo-voc-3c.meta',
    'threshold' : 0.3,
    'gpu' : 0.7 }
tfnet = TFNet( options )
color = (0, 255, 0) # bounding box color.

# This defines the area on the screen.
mon = {'top' : 10, 'left' : 10, 'width' : 1000, 'height' : 1000}
sct = mss()
previous_time = 0
while True :
    frame = sct.grab(mon)
    
    frame = np.array(frame)
    # image = image[ ::2, ::2, : ] # can be used to downgrade the input
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = tfnet.return_predict( frame )
    for result in results :
        tl = ( result['topleft']['x'], result['topleft']['y'] )
        br = ( result['bottomright']['x'], result['bottomright']['y'] )
        label = result['label']
        confidence = result['confidence']
        text = '{} : {:.0f}%'.format( label, confidence * 100 )
        frame = cv2.rectangle( frame, tl, br, color, 5 )
        frame = cv2.putText( frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2 )
    cv2.imshow ( 'frame', frame )
    if cv2.waitKey ( 1 ) & 0xff == ord( 'q' ) :
        cv2.destroyAllWindows()
        break
    txt1 = 'fps: %.1f' % ( 1./( time.time() - previous_time ))
    previous_time = time.time()
    