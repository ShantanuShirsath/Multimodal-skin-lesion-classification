# Multimodal Skin lesion Classification

Following project is a research project on use of Multimodal Attention based Model for Skin lesion Classification. The project was developed with the aim of implementing a multimodal Attention-based method that uses Radar signals from skin and image data to classify skin lesions and therefore eventually contribute in early detection of skin cancer.

## Project Details



**Data**: The data used for training a testing the model is part of Siegen University research data, collected from students and faculty. The data is in the form of,
- RAW data from FMCW Radar sensor used on skin lesion to capture the topographical feature.
- Images of skin lesion
Above data is not shared publically in the repository as per Siegen Univerity Policy.

**Model Architecture**: Some of the model architecture was adopted from the research paper - [Multimodal attention-based deep learning for Alzheimer’s disease diagnosis](https://academic.oup.com/jamia/article/29/12/2014/6712292?login=true) 
![Untitled Diagram drawio](https://github.com/ShantanuShirsath/Multimodal-skin-lesion-classification/assets/130396026/89fcffde-612f-406c-905f-61b60db00d2e)



**Model Training and Preprocessing**:  
- Preprocessing of Image data is done using a Convolutional Neural network and Radar data is done using the Fully connected network.
- This preprocessed Data is passed through Self Attention block to get an attention matrix for both Image and Radar data containing the attention scores for relevant features. This self-attention output is then passed through the Cross attention block to get the cross attention scores.
- Finally, the output of cross attention block is concatenated and passed through a fully connected layer for classification.

**Optimizers**:
A combination of SGD with Momentum and Adam optimizer is used to navigate through highly complex landscape of the model . Following is the loss curve for the model, with SGD for the initial 200 epochs and ADAM thereafter achieving a test accuracy of 94.5%
![loss curve](https://github.com/ShantanuShirsath/Multimodal-skin-lesion-classification/assets/130396026/fb6cf39c-c06d-494b-ae43-9d22818e3b77)
