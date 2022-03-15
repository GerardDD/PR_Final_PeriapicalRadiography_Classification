

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
- [Organization](#organization)
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

PENDING
*Include this section only if you chose to include ML in your project.*
* Describe how you trained your model, the results you obtained, and how you evaluated those results.
* Several configurations were used, performing first a simple NuSVC classifier to them, with the following results:
    - 

## Conclusion
PENDING
* Summarize your results. What do they mean?
* What can you say about your hypotheses?
* Interpret your findings in terms of the questions you try to answer.

## Future Work
PENDING
Address any questions you were unable to answer, or any next steps or future extensions to your project.

## Workflow

<img src="https://github.com/GerardDD/PR_Final_PeriapicalRadiography_Classification/blob/main/PR_Final_diagram.png" alt="Diagram workflow" />
PENDING
Outline the workflow you used in your project. What were the steps?
How did you test the accuracy of your analysis and/or machine learning algorithm?

## Organization
PENDING
How did you organize your work? Did you use any tools like a trello or kanban board?

What does your repository look like? Explain your folder and file structure.

## Links
PENDING
Include links to your repository, slides and trello/kanban board. Feel free to include any other links associated with your project.


[Repository](https://github.com/)  
[Slides](https://slides.com/)  
[Trello](https://trello.com/en)  
