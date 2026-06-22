[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/nXy_JSNi)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=18122255&assignment_repo_type=AssignmentRepo)

# My project: skin type image classifier

This is my Block C project (Year 1, ADS&AI). I built an image classifier that
sorts a face photo into skin types (normal, dry, oily) plus a separate class for
diseased skin. The idea is a skincare app that tells you your skin type from a photo.

## What I did

- Built my own image dataset. About 80% are real images (web scraping and an image
  downloader) and under 20% are AI generated, plus one Kaggle class for skin disease.
- Cleaned the data: converted everything to JPG, removed blurry images, checked the
  images are RGB, and counted images per class to keep it balanced.
- Trained four iterations: a basic MLP baseline, a CNN from scratch, the same CNN with
  data augmentation, and transfer learning with VGG16 and VGG19.
- Did error analysis on the misclassified images and a Responsible AI part on bias and
  fairness (Fairness Through Awareness), plus an A/B test of the app (W8 t-test notebook).

My role: this was an individual project, so the modelling, dataset and write-ups here
are mine. The peer review note in the Responsible AI notebook also includes feedback
from my reviewer (Barsam).

## Result

Best model was the VGG19 transfer learning one. Test accuracy was about 65% and the
best validation accuracy was around 68%. The model is very good at the skin disease
class but mixes up dry, normal and oily skin, since they look similar. Human level
performance I measured was 87%, so there is still a gap to close.

## How to run

1. Copy `.env.example` to `.env` and set `DATA_DIR` to the folder with your dataset.
   The dataset itself is not in the repo because it is too large (see `.gitignore`).
2. Install the packages from `Deliverables/requirements.txt` (it is a conda export).
3. Open `Deep-Learning-Template Mohammadali Finall (Ready to submit).ipynb` and run the
   cells in order. The notebook already has the saved outputs from my run.

---

# Block C - Data Modelling

In block A, you explored the foundations of artificial intelligence and data science by developing an interactive data visualization dashboard. In block B, you expanded your skills by analyzing a healthcare dataset and applying various preprocessing and machine learning methods to extract insights.

This block will focus on the __*Modeling*__ phase of the __*CRISP-DM*__ lifecycle. You will learn how to conduct market/consumer research and how to build transparent, interpretable, and fair deep-learning models for image classification. In addition, you will learn how to integrate these concepts for the development of user-centered applications.

## Creative Brief

The [Innovation Square](https://www.buas.nl/en/collaboration/innovation-square) is your client in this block. The Innovation Square is a dynamic hub at Breda University of Applied Sciences that integrates education, research, and industry. It's a place where collaboration and innovation connect education and practice-oriented research to activities in the relevant industries. They approached you - as an aspiring __Data Scientist__ - to apply your expertise in providing innovative data-driven solutions. In particular, they require your assistance in proposing and developing a creative and innovative application utilizing deep learning for image classification. The challenge is to identify a problem where image classification can provide significant business value and/or societal impact in any area or industry.

Therefore, the main objective of this project is to develop an image classification application using deep learning and your own image dataset. To this end, you will need to create a project proposal that touches upon the following topics:
- Market/consumer research and risk assessment;
- The design and evaluation of a transparent, interpretable (and fair) deep learning-based image classifier;
- The development of a user-centered prototype application for your image classifier.

The top 3 projects with the best business value will have the unique opportunity to present their results directly to the Innovation Square and a specially invited group of entrepreneurs from [BUas Startup Support](https://www.buas.nl/en/study/entrepreneurship) (like in the TV shows [Shark Tank](https://en.wikipedia.org/wiki/Shark_Tank) and [Dragons' Den](https://en.wikipedia.org/wiki/Dragons%27_Den_(British_TV_programme))), which can provide valuable insights and even support for further development of the projects, potentially transforming your academic projects into viable and standalone business ventures. This is more than just a project; it's a potential launchpad for your entrepreneurial journey! 

## Knowledge Modules

The ADS&AI program is structured into 8-week blocks. On Monday, Wednesday, and Thursday you work individually on the development of fundamental skills, which are needed to successfully complete the Creative brief. In *__DataLab__* (Mandatory! See [DataLab Attendance](https://adsai.buas.nl/General/DataLabAttendance.html), for more information), scheduled on Tuesdays and Fridays, you apply your knowledge to the Creative Brief by completing a list of tasks, which you can find in the [DataLab Tasks](https://adsai.buas.nl/Year1/BlockC/DataLabTasks.html).

The project of this block aims to develop an image classification application prototype. The block is centered around four *__knowledge modules__*:

- [Business Understanding](https://adsai.buas.nl/Study%20Content/Business%20Understanding/)
- [Responsible AI](https://adsai.buas.nl/Study%20Content/Responsible%20AI/)
- [Deep Learning](https://adsai.buas.nl/Study%20Content/Deep%20Learning/)
- [Human-Centered AI](https://adsai.buas.nl/Study%20Content/Human-Centered%20Artificial%20Intelligence/)

### 1. Business Understanding

For Business Understanding, you will conduct market research to identify a consumer related problem in a industry or company. Based on the stakeholder analysis and DAPS diagram, you will then create your first idea for an application aimed at solving a potential problem within that industry or company.

### 2. Responsible AI

For Responsible AI, you will perform an exploratory data analysis to uncover hidden biases in your custom image dataset or in the Imsitu dataset. In addition, after you have built and trained your image classification model, you will learn how to make it more transparent and interpretable by applying various explainable AI methods.

### 3. Deep Learning

For Deep Learning, you will explore various artificial neural network architectures and develop the skills to design and implement your own image classifier. Your model will demonstrate the feasibility of applying deep learning techniques to solve an image classification problem.

### 4. Human-Centered AI

For Human-Centered AI, you will design an application (wireframe prototype) based on your idea and DAPS diagram that incorporates your image classifier model. While designing the application, you also will conduct user tests (think-aloud study and A/B testing).

## Medal Challenges 

You are encouraged to get the best out of yourself. Therefore, within the ADS&AI program, we regularly allow you to push yourself further by giving you so-called bronze-silver-gold challenges. By achieving these, you can earn badges for your GitHub page, which mark excellent students: 

![badge](https://custom-icon-badges.herokuapp.com/badge/ADS&AI-1x-orange.svg?logo=bronzemedal) Build an **interactive explainable AI dashboard** that visualizes and interprets the predictions of your image classification model using techniques like Grad-CAM, LIME, or SHAP. The dashboard should allow users to upload new images, display the predictions, and show visual explanations highlighting the parts of the image that influenced the model's decision.

![badge](https://custom-icon-badges.herokuapp.com/badge/ADS&AI-1x-orange.svg?logo=silvermedal) Implement a **fully functional application** for the project, which includes the process of **deploying the image classification model on a server** and building a **functional client interface** to use the model.

![badge](https://custom-icon-badges.herokuapp.com/badge/ADS&AI-1x-orange.svg?logo=goldmedal) Get selected as one of the **top 3 projects** that will present their results to the Innovation Square and BUas Startup Support team. The selection of the best projects will be mainly based on business value, but keep in mind that other factors, such as model accuracy, interpretability, and interface design, also contribute to the viability of the project.
