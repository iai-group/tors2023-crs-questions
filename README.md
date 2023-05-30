<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**

- [Generating Usage-related Questions for Preference Elicitation in Conversational Recommender Systems](#generating-usage-related-questions-for-preference-elicitation-in-conversational-recommender-systems)
  - [Summary](#summary)
  - [Data](#data)
    - [Dataset Structure](#dataset-structure)
  - [Repository Structure](#repository-structure)
  - [Obtaining the dataset](#obtaining-the-dataset)
    - [Alternative](#alternative)
  - [Contact](#contact)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Generating Usage-related Questions for Preference Elicitation in Conversational Recommender Systems

## Summary

A key distinguishing feature of conversational recommender systems over traditional recommender systems is their ability to elicit user preferences using natural language.  Currently, the predominant approach to preference elicitation is to ask questions directly about items or item attributes. In this paper, we propose a novel approach to preference elicitation by asking implicit questions based on item usage.  As one of the main contributions of this work, we develop a multi-stage data annotation protocol using crowdsourcing, to create a high-quality labeled training dataset. Another main contribution is the development of four models for the question generation task: two template-based baseline models and two neural text-to-text models.  The template-based models use heuristically extracted common patterns found in the training data, while the neural models use the training data to learn to generate questions automatically.
## Data

We are not allowed to re-distribute the [Amazon review collection](https://nijianmo.github.io/amazon/index.html). Instead, we distribute our dataset with the review IDs (item ID and user ID) and sentence offsets. Additionally, we provide a script that derives the data collection we used from the original Amazon collection.

### Dataset Structure

The dataset contains 1100 reviews with matching implicit questions over 11 categories of products. It is split evenly into train (80%) and test (20%) files.
The test file additionally contains 15 questions in the `Birdfeeder` category that is not found in train dataset.

Since the sentences mentioning item usage were extracted heuristically, not all reviews in the dataset have valid questions associated with them. Those are marked as `n/a`.

The files have the following entries:

  - `id`: unique identifier
  - `category`: A category the item belongs to.
  - `question1`, `question2`, `question3`: Different variations of the usage question based on the review sentence obtained by crowdsourcing.
  - `paraphrase1`, `paraphrase2`: Question rewrites obtained by crowdsourcing where the input were `question1`, `question2`, and `question3` and not the review sentence.
  - `reviewerID`: Reviewer ID.
  - `asin`: product ID.
  - `start_index`: Start index of the extracted sentence.
  - `end_index`: End index of the extracted sentence.

Top 3 rows from the training dataset:

| Id         | category     | question1 |question2 |question3 |paraphrase1 |paraphrase2 |reviewerID|asin|start_index|end_index|
|--------------|-----------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| 307...3NL         | Tent     | Are you interested in a tent that is perfect for backpacking and biking? |do you want a tent who is perfect for backpacking and biking? |Do you want a tent that's perfect for backpacking and biking? |Can you use a tent that is great for both biking and hiking? |Do you want a comfortable cycling tent? |A2P8B5PMOIE7W|B00A8E2F88|0|34|
| 30E...6YK         | Walk-behind lawnmower     | Would you like a walk-behind lawnmower able to handle big yards? |Do you want a walk-behind lawnmower that can mow a big yard? |Are you looking for walk-behind lawnmower to mow a big yard? |Need a lawnmower that can mow a big yard? |How does a walk behind lawnmower to mow a big yard sound? |AEEI3GYQ5R0O5|B00Q2MGO32|80|139|
| 32T...84N         | Bike     | Are you looking for a bike that is great for commuting? |Would you like a bike that is good for commuting? |Do you want a bike that is great for commuting? |Are you interested in purchasing a bike that makes it easy for commuting? |Do you want a bike that can be used for commuting? |A2RLVLI4RIXPW8|B004Q3N0GI|0|84|

## Repository Structure

This repository is structured as follows:

  - `dataset/`: Train/test datasets collected via crowdsourcing. **NB! The files do not contain the original review text and extracted sentences.**
  - `outputs/`: The outputs of all four models.
  - `code/make_dataset.py`: A Python script for populating the dataset with original review text and extracted sentences.
  - `code/evaluate.py`: A Python script for evaluating the outputs of the models.
  - `code/models/`: Folder containing the code for the four models, both training and inference.

## Obtaining the dataset

To obtain the full dataset, use command:

```bash
python -m code.make_dataset --path <path_to_amazon_collection_folder>
```

The script expects files `Patio_Lawn_and_Garden.json.gz`, `Home_and_Kitchen.json.gz`, and `Sports_and_Outdoors.json.gz` to be present in the `<path_to_amazon_collection_folder>`. It parses through the files and populates the datasets with the original review and sentence texts.
The outputs are saved as `train_full.csv` and `test_full.csv` in the dataset folder.

### Alternative

Alternatively, you can email Ivica Kostric at <ivica.kostric@uis.no> or Krisztian Balog at <krisztian.balog@uis.no> to obtain the full dataset.

## Contact

Should you have any questions, please contact Ivica Kostric at <ivica.kostric@uis.no>.
