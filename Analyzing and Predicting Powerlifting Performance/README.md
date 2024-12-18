# Powerlifting Performance Prediction

## Overview

This project analyzes and predicts powerlifting performance using a dataset obtained from the [Open Powerlifting Database](https://www.kaggle.com/datasets/dansbecker/powerlifting-database). The main objective is to predict the total lift weight based on various features such as body weight, age, weight class, and sex of the lifter.

## Dataset

The dataset contains the following columns:

- **MeetID**: Identifier for the competition.
- **Name**: Name of the lifter.
- **Sex**: Gender of the lifter (Male/Female).
- **Equipment**: Type of equipment used (Raw, Equipped).
- **Age**: Age of the lifter.
- **Division**: Division category (e.g., Junior, Open).
- **BodyweightKg**: Bodyweight of the lifter in kilograms.
- **WeightClassKg**: Weight class of the lifter in kilograms.
- **Squat4Kg**: Squat weight attempted (4 attempts).
- **BestSquatKg**: Best squat weight lifted.
- **Bench4Kg**: Bench press weight attempted (4 attempts).
- **BestBenchKg**: Best bench press weight lifted.
- **Deadlift4Kg**: Deadlift weight attempted (4 attempts).
- **BestDeadliftKg**: Best deadlift weight lifted.
- **TotalKg**: Total weight lifted across all three lifts (squat + bench + deadlift).
- **Place**: Placement in the competition.
- **Wilks**: Wilks score (a measure of lifting performance).

## Installation

To run this project, make sure you have Python installed on your machine. You can download Python from [python.org](https://www.python.org/downloads/).

### Required Libraries

This project requires the following Python libraries:

- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install the required libraries using pip:

```bash
pip install pandas matplotlib seaborn scikit-learn
