# U-net-keras

A neural network for rough nodule detection

## image extraction

`img_extraction.py` is designed for extracting images from folders. There are two functions inside: _list_img_ & _clip_img_. _list_img_ is designed for extract images' address from folder. _clip_img_ is designed to fill each side of the image in order to enlarge the image to 512 x 512 and concatenate them to a 4D tensor(batch x Height x Width x Channel).

P.S. one can rewrite the _list_img_ function to fit their folder's structure

## image augmentation
`augment.py` is designed for image augmentation. 

## U-net
`net.py` is the main structure of U-net. `train.py` & `test.py` are used to train the weight and test the accuracy of the data.

P.S. `train.py` has a parse argument `-r` in order to continue the stopped training. Use like `python train.py -r`
