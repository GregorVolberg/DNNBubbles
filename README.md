# DNNBubbles
DNN for Bubbles data (NSSI and male / female faces)

works from Schlaubox

## Face data (with Vanessa M.)
Faces were shown on a screen with the task to classify the gender (male / female). 

### Strategies for data analysis
Learn bubbleized faces and classify full faces. The advantage is that per individual, the weights of the DNN layers could be identiftied and used in a layerwise back-propagation. resnet50 will be used.
Steps:
- Test resnet50. The support package can not be installed without admin privileges, try saving the layer information as '.mat file.
- does resnet50 learn from bubbles features? Of should a pre-trained network be used.
- export bubbleized faces with participants code and with trial number. This will be helpful later when the DNN ist trained, possibly on a subset of trials (2nd half) where specific features have been learned.

Alternatively, the net could learn from full faces and the bubbleized faces can be used for prediction. Will possibly not give the data to do a layerwise back-propagation.

