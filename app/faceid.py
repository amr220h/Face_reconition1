# import kivy dependencies first
import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.logger import Logger
from kivy.clock import Clock
from kivy.graphics.texture import Texture

# import other dependencies
import cv2
import numpy as np
import os
import tensorflow as tf
from layers import L1Dist


# Biuld app and layout 
class CamApp(App):
    def build(self):

        # main layout components
        self.wap_cam = Image(size_hint=(1, 0.8))
        self.button = Button(text='Verify',on_press =self.verify , size_hint=(1, 0.1))
        self.verification_lable = Label(text = 'Verification Status', size_hint=(1, 0.1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.wap_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_lable)

        self.model = tf.keras.models.load_model('siamese_model.h5', custom_objects={'L1Dist': L1Dist,'BinaryCrossentropy': tf.keras.losses.BinaryCrossentropy})
        # setp video capture
        self.capture = cv2.VideoCapture(0)
        
        
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        return layout
    def update(self,*args):
        ret,frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]
        # Flip the frame horizontally
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.wap_cam.texture = img_texture

    
    def preprocess(self, file_path):
        # read a image file
        byte_img = tf.io.read_file(file_path)
        # decode it into tenso with 3 channels (rgb)
        img = tf.image.decode_jpeg(byte_img)
        # preprocessing steps - rsizing the imge to 100*100
        img = tf.image.resize(img,(100,100))
        # scale image to be between 0 and1
        img = img/255.0
        return img
    
    # Bring over verification function

    def verify(self,*args):
        detection_threshold = 0.5
        verification_threshold = 0.5
        # Capture input image from our web cam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[120:120+250, 200:200+250, :]

        cv2.imwrite(SAVE_PATH, frame)
        # Build results array
        results = []
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            input_img = self.preprocess(os.path.join('application_data', 'input_image', 'input_image.jpg'))
            validation_img = self.preprocess(os.path.join('application_data', 'verification_images', image))
        
            # Make Predictions 
            result = self.model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)
        
        # Detection Threshold: Metric above which a prediciton is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold
        
        # Set verification status text
        self.verification_lable.text = 'verified' if verification == True else 'unverified'

        # Log the results
        Logger.info(results)
        Logger.info(f'Detection: {detection}, Verification: {verification}, Verified: {verified}')
        Logger.info(np.sum(np.array(results) > 0.2))
        Logger.info(np.sum(np.array(results) > 0.4))
        Logger.info(np.sum(np.array(results) > 0.6))
        Logger.info(np.sum(np.array(results) > 0.8))
        return results, verified



        
    




if __name__ == '__main__':
    CamApp().run()
