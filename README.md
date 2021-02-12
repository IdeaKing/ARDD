<h1> ARDD
  - Auto Retinal Disease Detection </h1>

<h3> ARDD is the winning webapp of the 2020 Congressional App Challenge for Virginia's 11th District. Created by Thomas Chia, Sreya Devarakonda, and Cindy Wu. </h3>

ARDD uses object detection and semantic image classification. The models used are: YOLOv3 and M-NET with Polar Transformation. The webapp is run on the Flask Micro WebFramework and can be run on a WSGI server. 

<h2> How does ARDD work? </h2>

ARDD can be separated into two stages: the object detection stage and the segmentation stage. The object detection stage will firstly detect the lesions and conditions within the retina. Then the segmentation stage will locate the optic disc region and segment the optic disc.

<h3> Stage 1 Example </h3>

Firstly the fundus image is entered:
![Image of input.](https://github.com/IdeaKing/aard/blob/main/uploads/original/5999712.jpg)

Then object detection is run on the image.

<h3> Stage 2 Example </h3>

Then the image is run on the segmentation model, returning something like this:
![Image of mask.](https://github.com/IdeaKing/aard/blob/main/uploads/masks/mask_5999712.jpg)

<h3> The Output </h3>

Lastly the output combines Stage 1 and Stage 2.

![Image of output.](https://github.com/IdeaKing/aard/blob/main/uploads/output/5999712.jpg)

<h2> Installing and Running ARDD </h2>

1. ` pip install -r requirements.txt `
2. ` python app.py `

*You may have to CD into the correct directory and enter the correct virtual environment/conda environment.*


More info to be posted soon. Please submit an issue if you have any question(s), I will do my best to answer them.
