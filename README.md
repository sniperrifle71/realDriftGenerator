# RealDriftGenerator

## Description

RealDriftGenerator is a concept drift generator designed to create custom concept drift on user-provided source temporal datasets. It offers higher complexity and temporal correlation compared to existing drift generators.
For details, please refer to this paper: "RealDriftGenerator: A Novel Approach to Generate Concept Drift in
Real World Scenario"

## Installation

Requires:
- NumPy 1.22.0
- Pandas 2.0.3

## Usage

First, create an instance of RealDriftGenerator by passing in the source temporal dataset (in DataFrame format). Then, apply concept drift by calling the RealDriftGenerator.reverseSlice method with a drift_dict that specifies the drift information. This method will return a DataFrame with the induced concept drift.

electricity_p700_w100_11000.csv and weather_p700_w100_l1000.csv are samples with 100 width concept drift in position 700 .

main.py is the sample code for RealDriftGenerator.

Currently, the project supports concept drift generation for classification problems only.

## License

This project is licensed under the MIT License.

## Authors

Lin Borong, email: borong.lin19@student.xjtlu.edu.cn

## Acknowledgments

Please cite paper "RealDriftGenerator: A Novel Approach to Generate Concept Drift in
Real World Scenario"

## Contact

Email: borong.lin19@student.xjtlu.edu.cn
For any program issues, please feel free to contact the author.
