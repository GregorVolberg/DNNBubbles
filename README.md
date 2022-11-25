# DNNBubbles
DNN for Bubbles data (NSSI and male / female faces)

works from Schlaubox

## Gender classification task (with Vanessa M.)
Faces were shown on a screen with the task to classify the gender (male / female). 

Data storage location is **/DNNBubbles/data/** with files
- BubblesFacesRaw.mat: Matlab structure with raw data for 28 participants
- fm_struct_npic_470x349.mat: Stimuli used for gender experiment, 6 scales x 470 x 349
- p5_struct_npic_470x349.mat: Stimuli used for emotion experiment, 6 scales x 470 x 349
- BubblesRawData_zwisch.mat: Matlab structure with raw data from >60 participants
- resnet50.mat: a pre-trained resnet50 network, obtained from net = resnet50


Image storage location is **/DNNBubbles/img/** with folders
- faces: raw male (n = 20) and female (n = 20) faces as exported from Kdyn. 
- GenderComposite: Bubbleized images in folders /female/correct, /female/incorrect, /male/correct, /male/incorrect. File names contain the subject code and the trial number, for later image selection. Call */DNNBubbles/img/get_GenderCompositeFace.m* for image construction.
- EmotionComposite: Bubbelized images  *(To do!!)*

Some tests with Gender classification data:
- *train_MaleFemale_res50.m*, trains resnet50 on 20 male / 20 female faces, result saved in ./data/trained_maleFemale_res50.mat
- *train_GenderComposite_res50.m*, trains resnet50 on bubbleized faces (correct responses only), result saved in ./data/trained_GenderComposite_res50.mat
- *train_GenderComposite_indiv_res50.m*, trains resnet50 on bubbleized faces (correct responses only), per subject, result saved in ./data/trained_GenderComposite_res50_[*vp-code*].mat

Ideas for analysis:
- train the network on full images, use relevane propagation for identifying relevant feature, compare with results of conventional bubble analysis.
- train network on bubbleiued images, per participants, use RDM and cluster analysis on layer weights. Which layer(s) are best suited to separate patient and control groups?
- 

## Emotion classification task (with Alex O.)



### Strategies for data analysis
Learn bubbleized faces and classify full faces. The advantage is that per individual, the weights of the DNN layers could be identiftied and used in a layerwise back-propagation. resnet50 will be used.
Steps:
- Test resnet50. The support package can not be installed without admin privileges, try saving the layer information as '.mat file. DONE: use resnet50.mat
- does resnet50 learn from bubbles features? Of should a pre-trained network be used. DONE: yes, learns from bubbleized images in gender classification task
- export bubbleized faces with participants code and with trial number. This will be helpful later when the DNN ist trained, possibly on a subset of trials (2nd half) where specific features have been learned.

Alternatively, the net could learn from full faces and the bubbleized faces can be used for prediction. Will possibly not give the data to do a layerwise back-propagation.

