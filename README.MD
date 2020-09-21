# White Blood Cell Segmentation Unet
A Tensorflow/Keras project to segment white blood cells from microscope images!

# How
I've implemented a model similar to Unet, which uses Inception to extract features, and then upsamples the feature map to output a mask. I'm working with binary masks (cell vs not cell), but the dataset supports trimaps as well, where you would have background, cytoplasm, and nucleus. For more information, please read the Medium article I wrote on the project: ARTICLE STILL BEING WRITTEN - LINK WILL BE HERE

# Usage
- Run 'train_model.py', which will train the model on the dataset, and produce 'wbc_segmentation_model.h5'
```shell
python3 train_model.py
```
- You know have the model in an h5 format! Feel free to use it wherever you'd like! To view the demo, run 'demo.py', or look at the 'demo.ipynb' notebook.
```shell
python3 demo.py
```


# Notes
- Please check out the Medium article about the project: LINK WILL BE HERE WHEN ARTICLE IS FINISHED
- There is not a lot of room for customization (unless you do it yourself), but if I have time, I may add args parser support to tweak parameters such as epochs, model architecture, etc
- You have the run 'train_model.py' to generate the model, before you demo/test it
- The 'original.ipynb' notebook is how I orginally trained the model, and the 'demo.ipynb' notebook contains a quick and simple demo, which shows the functionality of the model
- I may add support for saving transparent segmented images
- Project can be improved by teaching the model to segment background, cytoplasm, and nuclei, instead of just cell/not cell
- This approach can be used for a lot of other applications, such as segmenting faces from images
- Learn more about Unet: https://towardsdatascience.com/understanding-semantic-segmentation-with-unet-6be4f42d4b47
- Unet paper: https://arxiv.org/abs/1505.04597
- Please email me at 'zarif.azher@gmail.com', if you have any questions or comments!
- Dataset used: https://github.com/zxaoyou/segmentation_WBC
