# Yu-Gi-Oh! Card Classifier
This project was created to fulfill requirements for my MSc in Advanced Computing Science with Artificial Intelligence. It remains public for prospective employers and PhD studentships.

## What's here?
The master branch contains all the code that was used within the experiment as well as external utilities and outputs.

### Utils
Contains the original board states that were created for the purpose of the experiment. 
These board states were created with my own collection of cards and tried to test the limits of the pipeline.

The annotated board states are the output of the pipeline which is a bounding box around each card with the name of the card on top.

Card types holds the text files required for the fuzzy string search to find the name of the card based on the OCR check.

Models contain the saved weights of the CNN and resnet18 models used for classification tasks.
