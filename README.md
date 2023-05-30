# Generating Usage-related Questions for Preference Elicitation in Conversational Recommender Systems

## Summary

A key distinguishing feature of conversational recommender systems over traditional recommender systems is their ability to elicit user preferences using natural language.  Currently, the predominant approach to preference elicitation is to ask questions directly about items or item attributes.  
Users searching for recommendations may not have deep knowledge of the available options in a given domain. As such, they might not be aware of key attributes or desirable values for them.
However, in many settings, talking about the *planned use* of items does not present any difficulties, even for those that are new to a domain.  In this paper, we propose a novel approach to preference elicitation by asking implicit questions based on item usage.   As one of the main contributions of this work, we develop a multi-stage data annotation protocol using crowdsourcing, to create a high-quality labeled training dataset.
Another main contribution is the development of four models for the question generation task: two template-based baseline models and two neural text-to-text models.  The template-based models use heuristically extracted common patterns found in the training data, while the neural models use the training data to learn to generate questions automatically.
Using common metrics from machine translation for automatic evaluation, we show that our approaches are effective in generating elicitation questions, even with limited training data.  
We further employ human evaluation for comparing the generated questions using both pointwise and pairwise evaluation designs. We find that the human evaluation results are consistent with the automatic ones, allowing us to draw conclusions about the quality of the generated questions with certainty.  Finally, we provide a detailed analysis of cases where the models show their limitations.

## Repository Structure

This repository is structured as follows:

  - `dataset/`: Train/test datasets collected via crowdsourcing. **NB! The files do not contain the original review text and extracted sentences.**
  - `outputs/`: The outputs of all four models.
  - `code/make_dataset.py`: A Python script for populating the dataset with original review text and extracted sentences.
  - `code/evaluate.py`: A Python script for evaluating the outputs of the models.
  - `code/models/`: Folder containing the code for the four models, both training and inference.

## Obtaining the dataset

To obtain the full dataset use command:

```
python -m code.make_dataset --path <path_to_amazon_collection_folder>
```

The script expects files `Patio_Lawn_and_Garden.json.gz`, `Home_and_Kitchen.json.gz`, and `Sports_and_Outdoors.json.gz` to be present in the `<path_to_amazon_collection_folder>`. It parses through the files and populates the datasets with the original review and sentence texts.
The outputs are saved as `train_full.csv` and `test_full.csv` in the dataset folder.

### Alternative

Alternatively, you can email Ivica Kostric at <ivica.kostric@uis.no> or Krisztian Balog at <krisztian.balog@uis.no> to obtain the full dataset.

## Contact

Should you have any questions, please contact Ivica Kostric at <ivica.kostric@uis.no>.
