# Hack The Summer
## AIMLC, IIT Delhi

## Optical character recognition for simple formulas

My submission includes an AI training program, train.py, that can train an AI with decent accuracy without any external datasets, pretrained AI or human intervention, you can just give the path of the dataset and the annotations file and it will train the ai for you. It can be any dataset of the format in the porblem statement given that it's big enough, it can even be in an intirely different language provided that the symbols are distinct enough. Infact you can try it out by downloading train.py and running python3 train.py /path/to/data /path/to/annotations/file and it'll train an AI with about 97% accuracy in 20 to 70 minutes depending upon your hardware.

## Methodology used

At first sight it looks nearly impossibe to train an AI just by the results of the formulas with just an MNIST-like AI (k-means clustering or something similar might work but I'm constraining myself to only what we have been taught in lectures). However upon further inspection you'll start noticing that there is some information you can know for certain from the data given to you. The trick is to look for the results that can be made in one way only and then using them to train an AI that can recognize those digits, which you can then use to recognize new digits. This goes on in a loop till you have datasets for every digit. I'll be explaining exactly what i did later on but first we need to make sure that it's easy for the AI to actually recognize digits, which is where image preprocessing and convolutions come in.

### Image Preprocessing

Recognizing symbols is hard for a machine, but what makes it even harder is unnecessary variations in data that only confuse the AI. Namely the position and sie of the individual symbols. These attributes don't contribute anything to the recognition of the numbers and only cause inefficiencies, so our first step should be to eliminate them. I have implemented a simple function, cropAndProcessImage that takes an image, divides it into three parts, crops each part individually such that the digit is in the center and is scaled properly and then resizes them to 28 by 28 pixels (yes the size is inspired by the MNIST dataset). The reduction in size makes it so that the AI can easily process th data and conterintuetively, it doesn't even cause a huge loss in the overall "quality" of the image. This is because the givven images are completely black and white with no grays, so converting them into images with a 0-255 pixel color variation with lower size doesn't lead to much difference. You can think of it as the edges being marked by gray colors. An example of the image preprocessing is given below.


