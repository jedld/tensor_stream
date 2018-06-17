# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2018-06-17
### Added
- Allow float64, int16 datatypes for the OpenCL evaluator
- Various bug fixes
- Tweaked data type checking to be more tensorflow like
- initial work on docs
- [OpenCL] tweaked best device detector
- [NEW OP] Added op softmax_cross_entropy_with_logits
- [NEW OP] check_numerics
### Fixes
- fixed/improved softmax behavior


## [0.3.0] - 2018-06-05
### Added
- hardware acceleration using OpenCL
- working nearest neighbor sample (use opencl evaluator for best performance)

## [0.2.0] - 2018-05-27
### Added
- working logistic regression sample
- reworked auto differentiation, fix a number of bugs related to auto differentiation, smaller derivative programs
- alpha support for saving to pbtext format, added graphml generation
- significant number of ops added
- ops that support broadcasting now work better