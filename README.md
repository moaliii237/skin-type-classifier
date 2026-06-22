# Skin type image classifier

This is my Year 1 Block C project for the ADS&AI program at BUAS. I built an image
classifier that looks at a photo and sorts the skin into four classes: dry, normal,
oily, and skin disease. The idea behind it is a skincare app that could tell you your
skin type from a single photo. The hard part is that the three healthy skin types look
very similar to each other, so this is not an easy 4-class problem, and my numbers show
that.

I did this project on my own, so the dataset, the modelling, and the write-ups here are
all mine. The only outside input is a peer review note in the Responsible AI part, which
came from my reviewer (Barsam, 244141).

## What is in here

- A custom image dataset that I collected and cleaned myself (not committed, see below).
- A deep learning notebook with four model iterations, from a simple baseline up to
  transfer learning, plus error analysis.
- A Responsible AI notebook where I look at bias in my own dataset and pick a fairness
  method.
- An A/B usability test of the app idea (the survey results and analysis are in the
  `A&B Test` folder).

## How I built the dataset

I had to make my own dataset, there was no ready one for this. I ended up with four
classes and a mix of sources:

- About 80% real images. Most came from an image downloader tool, some from web
  scraping (slower, and I had to filter the results by hand). I also pulled in a Kaggle
  set for the skin disease class so I had more variety there.
- Under 20% AI generated images, made with ComfyUI through its API. I wrote 30 prompts
  per class. I kept these limited on purpose, because I did not want the model learning
  only from fake skin that misses the small natural details.

After collecting everything I cleaned it in the notebook (`Deep-Learning-Template
Mohammadali Finall (Ready to submit).ipynb`). The steps, all with OpenCV, were:

1. Convert every image to JPG with `cv2`, so the whole set is one consistent format.
2. Remove blurry images. I used the Laplacian variance trick: run a Laplacian filter on
   the grayscale image, take the variance, and if it is below a threshold of 100 the
   image is too blurry, so I dropped it. Blurry images just confuse the model.
3. Check every image is real RGB (3 channels), because mixed channels cause problems.
4. Remove exact duplicates by hashing each file with MD5 and deleting any hash I had
   already seen. This matters because the downloader and scraper grab the same image
   more than once.
5. Count images per class to check the classes were not badly unbalanced.

Then I split the data 70/20/10 into train, validation, and test with scikit-learn
`train_test_split` and a fixed `random_state=42` so the split is reproducible. Note the
split script moves the files, so it is meant to run only once.

## My modelling approach

I worked through four iterations on purpose, so I could see what actually helped instead
of jumping straight to a big model.

**Baselines first.** Random guessing on four classes is 25%. I also measured human level
performance by giving 5 people a 40 question quiz (10 per class). They got 26 wrong in
total, which works out to about 87%. So 25% is the floor and 87% is the target.

**Iteration 0, MLP baseline.** A plain fully connected network with the Keras Sequential
API: Flatten on the 256x256 RGB image, then dense layers of 256, 128, and 64 units with
ReLU, Batch Normalization, and Dropout (0.4 then 0.2). Output is 4 softmax neurons. Adam
at learning rate 0.0003, categorical crossentropy, and EarlyStopping with patience 3.
This is weak by design, an MLP throws away the spatial structure of the image, but it
gives a starting point.

**Iteration 1, CNN from scratch.** Several convolutional blocks, each one Conv2D plus
Batch Normalization plus MaxPooling, with filters growing from 32 up to 512, then Dropout
0.5 and dense layers into a 4-way softmax. Batch size 16 (smaller batch, more frequent
weight updates, more stable for me), EarlyStopping with patience 5 and
`restore_best_weights`. This got about 59% test accuracy (58.57% to be exact). The
confusion matrix already showed the main story: it nails skin disease and confuses the
three healthy types.

**Iteration 2, CNN with augmentation.** To get more variety I augmented the training set
with `ImageDataGenerator`: rotation 20 degrees, width/height shift 0.1, zoom 0.2,
brightness 0.8 to 1.2, and horizontal flip. I made 3 augmented copies per image and saved
them to a new `aug_train` folder, which grew the training set to 4,151 images (validation
stayed at 415 and test at 210, both only rescaled, never augmented). This CNN had around
13 million parameters. Honestly the augmentation did not help my numbers here, test
accuracy was 56.19% and validation 58.07%, slightly below the plain CNN, which taught me
that augmentation is not a free win when the classes overlap this much.

**Iteration 3, transfer learning with VGG16.** Resized to 224x224 for the pretrained
input. I used VGG16 (ImageNet weights) as a frozen feature extractor, then fine tuned only
the last 6 layers, with my own dense head (512, 256, 128) plus Batch Normalization and
Dropout. Adam at a very low learning rate (0.00003) so the pretrained features do not get
wrecked, class weights to fight the imbalance, and two callbacks, EarlyStopping and
ReduceLROnPlateau. This was my best model: 64.76% test accuracy (loss 0.766) and best
validation 68.67%. Skin disease had recall 0.98 and precision 0.96. The macro and weighted
F1 were both around 0.64.

**Iteration 4, VGG19.** Same idea with VGG19 as the base, `include_top=False`, dense head
of 512, 256, 128 with heavier dropout (0.6, 0.5, 0.5), Adam at 0.00005, plus EarlyStopping
and ReduceLROnPlateau. Best validation accuracy 67.95% at epoch 29. Macro and weighted F1
both 0.66. Skin disease again came out strongest (precision 0.90, recall 0.97), and dry,
normal, and oily skin sat lower with F1 between 0.52 and 0.60.

**Error analysis.** I pulled out the misclassified images and grouped them. Most mistakes
fall into a few buckets: the three healthy skin types look alike (especially normal vs
dry), some images were probably mislabeled, and some were blurry or low contrast so the
skin texture was not clear. My proposed fixes were to recheck the labels, collect more
clean images for the confusing classes, and add contrast enhancement.

## Responsible AI

In `Responsible-AI-Template.ipynb` I look at bias in my own dataset before trusting the
model. The biases I flagged: the AI generated portion can miss real skin detail, there is
likely demographic and skin tone imbalance, the web scraped images vary a lot in lighting
and quality, and mislabeling is dangerous in anything skin or medical related. I chose
skin tone as the sensitive attribute and picked Fairness Through Awareness as the fairness
method, the idea being to acknowledge the skin tone differences and train fairly across
groups rather than pretend all images are equal. This notebook also holds the infographic
peer review from my reviewer.

## A/B usability test

For the Human-Centered AI side I ran an A/B test on two versions of the app design and
collected Likert scale survey answers. I used SciPy for the statistics: a Shapiro-Wilk
test to check whether the responses were normally distributed, Levene's test for equal
variance, and a t-test to compare the two versions. The survey material and the analysis
are in the `A&B Test` folder (versions A and B).

## Results, stated honestly

| Model | Test accuracy |
| --- | --- |
| Random guess (4 classes) | 25% |
| MLP baseline | low, baseline only |
| CNN from scratch | ~59% (58.57%) |
| CNN + augmentation | 56.19% |
| VGG16 transfer learning | 64.76% (best) |
| VGG19 transfer learning | best val 67.95% |
| Human level performance | ~87% |

My best model lands around 65% on the test set. That is clearly better than the 25%
random baseline, but it is well below the 87% humans got, so the project is not
production ready. The reason is not really the architecture, it is the problem and the
data: dry, normal, and oily skin look very close to each other even to a person, the
dataset is not huge, and part of it is AI generated or possibly mislabeled. The one thing
every model agreed on was that skin disease is easy to separate from healthy skin. If I
came back to this, I would spend the time on the dataset (better labels, more real images,
better balance) before touching the model again.

## Tech stack

- Python 3.9
- TensorFlow / Keras 2.11 for the models
- OpenCV (`opencv-python` 4.11) for the image cleaning (format conversion, blur filter,
  RGB check)
- scikit-learn 1.6.1 for the train/val/test split and the classification reports
- scikit-image 0.24 for image utilities
- SciPy 1.13.1 for the A/B test statistics (Shapiro-Wilk, Levene, t-test)
- NumPy, Matplotlib, Pillow
- xplique was installed for explainable AI experiments

## How to run

The dataset and the trained model files (`.h5`, `.keras`) are not in this repo, they are
too large and the data is kept off GitHub (see `.gitignore`). You bring your own dataset
folder with one subfolder per class.

1. Copy `.env.example` to `.env` and set `DATA_DIR` to the folder that holds your dataset.
   If you want to regenerate the AI images, also set the `COMFYUI_*` values.
2. Create the environment from the conda export:
   ```
   conda create --name skin --file "Deliverables/requirements.txt"
   conda activate skin
   ```
3. Open `Deep-Learning-Template Mohammadali Finall (Ready to submit).ipynb` and run the
   cells in order. The notebook keeps the saved outputs from my run, so you can read the
   results without retraining. Do not re-run the dataset split cell, it moves files and is
   meant to run only once.

A GPU helps a lot for the CNN and VGG iterations. On CPU the transfer learning models are
slow to train.

## Key files

```
Deep-Learning-Template Mohammadali Finall (Ready to submit).ipynb   main notebook: data cleaning + 4 model iterations + error analysis
Responsible-AI-Template.ipynb                                       bias analysis and fairness method
A&B Test/                                                           A/B usability test (versions A and B, survey + analysis)
Deliverables/requirements.txt                                       conda environment export
.env.example                                                        config template (DATA_DIR, ComfyUI server)
```
