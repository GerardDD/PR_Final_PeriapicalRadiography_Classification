

<img src="https://bit.ly/2VnXWr2" alt="Ironhack Logo" width="100"/>

# Periapical Radiography classification
*[Gerard Domenech Domingo]*

*[Data part time SEP-21]*

## Content
- [Project Description](#project-description)
- [Hypotheses / Questions](#hypotheses-questions)
- [Dataset](#dataset)
- [Cleaning](#cleaning)
- [Analysis](#analysis)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Workflow](#workflow)
- [Links](#links)

## Project Description 

Periapical radiographies are x-rays images that show full teeth: from the root to the crown.

In this project we will try to discern a healthy tooth from and unhealthy one, by applying CNN and image processing techniques to classify x-rays (focusing on the root), a task that has been proven to be difficult even for seasoned odontologists.

## Goal

Our goal is to design an algorithm that could help on identifing unhealthy roots, and thus be able to help a profesional odontologist to make choices easily.


## Dataset

Dataset is obtained from real individuals, although no more information can be disclosed due to LOPD laws.

The dataset contains 390 x-ray radiographies: 198 healthy, 192 unhealthy teeth.
Additional categorical information such as age and sex is provided partially.
 
* If the question cannot be answered with the available data, why not? What data would you need to answer it better? PENDING 

## Cleaning

- Manually extracting sex and age information from the images themself
- Erasing private information from images using OpenCV library
- Cropped images to focus on the important parts using OpenCV library


## Analysis
 PENDING

* First I noted that xrays images have different orientation, so I manually rotated them so they all have the same
* Then I tried different configurations of base dataset:  censored images, u uncesored images, cropped images, uncropped images.
* I used two different pretrained models to preprocess images: VGG16 and VGG19
* Every configuration pretrained was dumped on individual pickles so they can be used in the near future if needed
* Then I used lazypredict library to run a batch of vanilla models, thus acting as first model
selection process
* The top 5 models where then gridsearched for every configuration.



## Model Training and Evaluation

* Describe how you trained your model, the results you obtained, and how you evaluated those results.
* Several configurations were used using the vectorized VGG16 images, performing first a simple NuSVC classifier to them. Results on training score were acceptable, but test score were mostly poor.
* Then, vectorized VGG19 images were used, and the results were more promising

<img src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/Models_table.png" alt="Models table" />

* ROC curves were used to evalute models:

<img src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/ROC_randomforest_vgg16.png" alt="VGG16 ROC" />

<img src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/ROC_randomforest_vgg19.png" alt="VGG19 ROC" />

* Learning curves were evaluated too:

- VGG16 rotated

<img src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/rotated_images_Lcurves.png" alt="VGG16 rotated" />

- VGG16 cropped

<img src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/croped_images_Lcurves.png" alt="VGG16 cropped" />

- VGG16 uncens

<img src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/uncensVGG16_images_Lcurves.png" alt="VGG16 uncens" />

- VGG19 uncens

<img src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/uncensVGG19_images_Lcurves.png" alt="VGG19 uncens" />

* Unfortunately, upon testing the model on a set of validation images, the result was bad, not detecting a single unhealthy tooth

* Further analysis of the dataset, revealed that xrays belonging to the same person were misleading the model as ther were too similar and "easy" to train and classify.

* I trained new models but this time manually filtering the dataset to not include xrays of the same person

* This time results train and test scores were lower, but more according to the reality:

<img 
src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/ROC_randomforest_vgg19_selected_female.png" alt="VGG 19 selected female" />

<img 
src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/ROC_randomforest_selected_vgg19.png" alt="VGG 19 selected" />

<img 
src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/ROC_randomforest_vgg19_selected_male.png" alt="VGG 19 selected male" />


* After trying classical Machine Learning models, I tried xgboosting and Adaboosting. Results are slightly better:

<img 
src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/ROC_Boost_vgg19_selected_as.png" alt="VGG 19 xgboost selected" />

<img 
src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/Modelo/ROC_Ada_vgg19_selected_as.png" alt="VGG 19 Adaboost selected" />


When trying with a validation set of images (not used neither in training nor testing sets),
I obtained the following results with xgboost:

Columna1 | image name | result | Real |Check
---------|------------|--------|------|-----
|0|SIN_sexU_age35__.JPG|1|0|FALSO
|1|CON_age35_sexU___.JPG|1|1|VERDADERO
|2|CON_age999_sexM.JPG|1|1|VERDADERO
|3|SIN_sexF_age47_.JPG|0|0|VERDADERO
|4|CON_age38_sexF__.JPG|1|1|VERDADERO
|5|SIN_sexF_age47___.JPG|0|0|VERDADERO
|6|SIN_sexU_age35_.JPG|0|0|VERDADERO
|7|CON_age38_sexF.JPG|1|1|VERDADERO
|8|SIN_sexF_age47____.JPG|1|0|FALSO
|9|CON_age35_sexU.JPG|1|1| VERDADERO
|10|SIN_sexF_age47__.JPG|1|0|FALSO
|11|CON_age35_sexU__.JPG|1|1|VERDADERO


## Conclusion
PENDING

* Current model is not accurate enough to be used as a proper tool of classification and diagnosis,
however it may be used as an additional source for Kappa scoring using by odontolgists (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/).



## Future Work
PENDING
Address any questions you were unable to answer, or any next steps or future extensions to your project.
* It is clear that there is still a lot of room for improvement. There are different paths that can be further explored to achieve a higher accuracy:
1) Obtain more and better data: The dataset was relatively small so it is possible that was not enough for the model to be trained properly.
2) Create a new CNN: Instead of using VGG19 transfer learning, created a new specific neural network from scratch so it can be specific for our case.

## Workflow

<img src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/PR_Final_diagram.png" alt="Diagram workflow" />


## Links
PENDING
Include links to your repository, slides and trello/kanban board. Feel free to include any other links associated with your project.


[Repository](https://github.com/)  
[Slides](https://slides.com/)  
[Trello](https://trello.com/en)  
