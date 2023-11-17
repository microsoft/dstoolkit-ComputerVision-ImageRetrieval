import os, sys
import json
import yaml
import time
import requests
from flickrapi import FlickrAPI
import azure.ai.vision as sdk
import numpy as np
import faiss
import pickle
import matplotlib.pyplot as plt

class AzureImageRetrieval():
    def __init__(self,
                 config_file: str) -> None:
        ## Prepare config file
        self.config_file = config_file
        self.load_config()
        ## image configuration
        self.image_folder = self.config['image']['folder']
        ## Configuration in retrieving images in Flickr
        self.API_KEY = self.config['flickr']['API_KEY']
        self.API_SECRET = self.config['flickr']['API_SECRET']
        self.NUMBER_OF_IMAGES = self.config['flickr']['NUMBER_OF_IMAGES']
        self.NUMBER_PROCESS_IMAGES = self.config['flickr']['NUMBER_PROCESS_IMAGES']
        ## Configuaration in Azure
        self.CV_ENDPOINT = self.config['Azure']['ENDPOINT']
        self.CV_KEY = self.config['Azure']['KEY']
        self.vectorizeImageEndpoint = self.CV_ENDPOINT + '/computervision/retrieval:vectorizeImage?api-version=2023-02-01-preview&modelVersion=latest'
        self.vectorizeTextEndpoint = self.CV_ENDPOINT + '/computervision/retrieval:vectorizeText?api-version=2023-02-01-preview&modelVersion=latest'
        self.headers = {
            "Content-Type": "application/octet-stream",  # binary data in sending API
            "Ocp-Apim-Subscription-Key": self.CV_KEY
        }
        self.headersForVectorizeText = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.CV_KEY
        }
        ## metadata store
        self.nameVectors = self.config['metadata']['vectors_name']
        self.vectors = dict()
        ## initialization of image SDK
        self.analysis_options = sdk.ImageAnalysisOptions()
        self.service_options = sdk.VisionServiceOptions(self.CV_ENDPOINT,
                                                        self.CV_KEY)
        ## search index with Faiss format
        self.dimension = self.config['faiss']['dimension']
        self.index_flat_l2 = faiss.IndexFlatL2(self.dimension)
        self.filename_index = self.config['faiss']['filename']
        self.top_N = self.config['faiss']['top_N']
        ## images
        self.images = []
        self.num_cols = self.config['display']['num_cols']

    def load_config(self):
        '''
        Load and extract config yml file.
        '''
        try:
            with open(self.config_file) as file:
                self.config = yaml.safe_load(file)
        except Exception as e:
            print(e)
            raise

    def downloadImages(self):
        try:
            # Initialize FlickrAPI
            flickr = FlickrAPI(self.API_KEY, self.API_SECRET, format='parsed-json')

            # Search images under Creative Commons license
            photos = flickr.photos.search(license='1,2,3,4,5,6', per_page=self.NUMBER_OF_IMAGES)  # 1-6 shows creative commons in Flickr

            # Create directory for downloaded images
            if not os.path.exists('downloaded_images'):
                os.makedirs('downloaded_images')

            # Download image 1 by 1
            for i, photo in enumerate(photos['photos']['photo']):
                if i % 10 == 0:
                    print(f'{i} / {self.NUMBER_OF_IMAGES} images downloaded')
                time.sleep(2)
                photo_id = photo['id']
                farm_id = photo['farm']
                server_id = photo['server']
                secret = photo['secret']
                
                # Populate URL for downloading
                url = f'https://farm{farm_id}.staticflickr.com/{server_id}/{photo_id}_{secret}.jpg'
                
                # Download image
                response = requests.get(url, stream=True)
                with open(f'downloaded_images/{photo_id}.jpg', 'wb') as file:
                    for chunk in response.iter_content(chunk_size=8192):
                        file.write(chunk)

            print("Complete download!")
        except Exception as e:
            print(e)
            raise

    def getVector(self, 
                  image: str) -> np.array:
        '''
        Get vector with Vector Image API for one image
        '''
        try:
            with open(image, mode="rb") as f:
                image_bin = f.read()
            response = requests.post(self.vectorizeImageEndpoint
                                     ,headers=self.headers
                                     ,data=image_bin)
            return np.array(response.json()['vector'], dtype='float32')
        except Exception as e:
            print(e)
            raise

    def getVectorWithText(self,
                          query_text: str) -> np.array:
        '''
        Convert query text to vector with computer vision API        
        '''
        try:
            data = {
                'text': query_text
            }
            response = requests.post(self.vectorizeTextEndpoint
                                     ,headers=self.headersForVectorizeText
                                     ,data=json.dumps(data))
            return np.array(response.json()['vector'], dtype='float32').reshape(1, -1)
        except Exception as e:
            print(e)
            raise

    def searchIndexWithText(self,
                              query_text:str) -> list:
        '''
        Search with query text
        '''
        try:
            query_vector = self.getVectorWithText(query_text=query_text)
            ## Search
            return self.index_flat_l2.search(query_vector, self.top_N)
        except Exception as e:
            print(e)
            raise

    def getImageProperties(self,
                   image:str) -> str:
        try:
            self.analysis_options.features = (
                sdk.ImageAnalysisFeature.CROP_SUGGESTIONS |
                sdk.ImageAnalysisFeature.CAPTION |
                sdk.ImageAnalysisFeature.DENSE_CAPTIONS |
                sdk.ImageAnalysisFeature.OBJECTS |
                sdk.ImageAnalysisFeature.PEOPLE |
                sdk.ImageAnalysisFeature.TEXT |
                sdk.ImageAnalysisFeature.TAGS
            )
            self.analysis_options.language = "en"
            self.analysis_options.gender_neutral_caption = True
            vision_source = sdk.VisionSource(filename=image)
            image_analyzer = sdk.ImageAnalyzer(self.service_options
                                               , vision_source
                                               , self.analysis_options)
            ## Analyze image
            return image_analyzer.analyze()
        except Exception as e:
            print(e)
            raise

    def getVectorFromImages(self):
        '''
        - input:
            folder: Local folder name including images
        '''
        try:
            ## Get image file name
            images = [image for image in os.listdir(self.image_folder) if image.endswith('.jpg')]
            print(f'Target images: {len(images)}')
            ## Analyze each image with API
            for i, image in enumerate(images[:self.NUMBER_PROCESS_IMAGES]):
                ## Set image path
                image_path = os.path.join(self.image_folder, image)
                ## Get vector for image
                vector = self.getVector(image=image_path).reshape(1, -1)
                print(image, vector)
                ## Analyze with Azure Vision API
                analyzed_result = self.getImageProperties(image=image_path)
                try:
                    caption_content = analyzed_result.caption.content
                except:
                    caption_content = None

                ## Store the results
                self.vectors[image] = {}
                self.vectors[image]['index'] = i
                self.vectors[image]['vector'] = vector
                self.vectors[image]['caption'] = caption_content
                time.sleep(3)
                ## Store vector in search index
                self.index_flat_l2.add(vector)
        except Exception as e:
            print(e)
            raise

    def storeObj(self) -> None:
        '''
        Store vectors as pkl file
        '''
        try:
            with open(self.nameVectors, "wb") as f:
                pickle.dump(self.vectors, f)
        except Exception as e:
            print(e)
            raise

    def storeIndex(self) -> None:
        '''
        Store Index as pkl file
        '''
        try:
            with open(self.filename_index, 'wb') as f:
                pickle.dump(self.index_flat_l2, f)
        except Exception as e:
            print(e)
            raise

    def loadIndex(self) -> None:
        try:
            with open(self.filename_index, 'rb') as f:
                self.index_flat_l2 = pickle.load(f)
        except Exception as e:
            print(e)
            raise

    def loadObj(self) -> list:
        try:
            with open(self.nameVectors, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(e)
            raise

    def sortImages(self,
                   query_text: str):
        '''
        Convert index number to image file names
        '''
        try:
            self.images = []
            D, I = self.searchIndexWithText(query_text=query_text)
            for i in I[0]:
                image_name = {k for (k, v) in self.vectors.items() if v['index'] == i}.pop()
                self.images.append(image_name)
        except Exception as e:
            print(e)
            raise

    def displayWithText(self,
                        query_text: str) -> None:
        '''
        With query_text, display images with captions
        '''
        try:
            image_directory = './downloaded_images/'
            ## Get images
            self.sortImages(query_text=query_text)
            num_images = len(self.images)
            num_rows = num_images  # 行数
            num_cols = self.num_cols

            # configuration of subplot
            fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 6))

            for i, ax in enumerate(axes.flat):
                if i < num_images:
                    ## Show image
                    image_path = os.path.join(image_directory, self.images[i])
                    image = plt.imread(image_path)
                    ax.imshow(image)
                    ## Get caption from API
                    caption = {v['caption'] for (k, v) in self.vectors.items() if k == self.images[i]}.pop()
                    ax.set_title(caption)
                ax.axis('off')

            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(e)
            raise


