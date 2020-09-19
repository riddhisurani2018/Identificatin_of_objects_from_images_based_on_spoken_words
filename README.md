# Identificatin_of_objects_from_images_based_on_spoken_words

1. create_small_dataset.ipynb
    "Selects random images from COCO dataset and creates a small dataset consisting of 3 folders: "train", "test", "val""
    "train": Contains 500 images
    "test": Contains 100 images
    "val": Contains 100 images

2. Create .pkl files with information about object class and bounding box
    
3. get_object_audio_pairs.ipynb
    "Iterates over objects in each image and object mentions in audio caption and map image objects to auido objects"
        i.e. If I found that there is a TV in image and there is TV spoken in audio then I will add a line to the ".tsv" file with information about location of TV in image and audio

4. create_object_audio_files_true.ipynb
    "Iterate over each sample from ".tsv" files crop the image object create spectrogram for audio slice (object mention)"
    
5. extract_img_feats.ipynb
    "Iterate over all the croppings of objects and extract their features"
    
6. train.ipynb
    "Train the model using ResNet152 features of object croppings and spectrogram images"

7. evaluate.ipynb
    "Test the model performance in recognising related ojects and audios using "test" split"
    
8. demo.ipynb
    "Given an image name and an empty folder, finds objects in image and audio used the trained model to determine their relatedness scores and selects good object and audio mappings greedily"
        Stores the output audio and image croppings in given output dir
    run:
        python demo.py image_name output_dir
