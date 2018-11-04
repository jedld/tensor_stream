# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.9.5] - 2018-11-05
- [NEW OP] assert_equal, relu6
- [TRAINING] learning_rate_decay, dropout
- [BUG FIX] argmin, argmax now works properly
- [BUG FIX] shape inference fixes

## [0.9.2] - 2018-10-19
- Add profiling support
- Make sure sparse ruby arrays are caught
- Enhancements to support multi-gpu execution

## [0.9.1] - 2018-10-19
- Bug fix release

## [0.9.0] - 2018-10-05
- Bug fix release for OpenCL gem

## [0.8.6] - 2018-09-11

### Added
- [TRAINING] Added RMSPropOptimizer, AdagradOptimizer
- [NEW OP] shape_n, sparse_softmax_cross_entropy_with_logits, split, unstack
- Added RNN sample

### Fixes
- Fixed gradient computation when passing an array of tensors to a function
- Added gradients for various other ops

## [0.8.5] - 2018-09-06

### Added
- [TRAINING] Added AdadeltaOptimizer
- [NEW OP] squeeze, encode_png, decode_png

### Others
- The OpenCL evaluator has now been decoupled and is not on its own gem (tensor_stream-opencl)

## [0.8.1] - 2018-08-30
- [TRAINING] Added AdamOptimizer

## [0.8.0] - 2018-08-29
### Added
- [TRAINING] Added new supported optimizer, MomentumOptimizer loosely based on tensorflow's implementation (with nesterov support)
- [NEW OP] fill, stack, atan, cumprod, gather, invert_permutation, setdiff1d

### Fixes
- Fixed device delegator where it does not pick the correct evaluator to use in some cases
- [GRADIENTS] Properly implement gradient computation for prod, tile, transpose
- Fixed gradient computation for softmax_cross_entropy_with_logits_v2 (now based on tensorflow's implementation)

## [0.7.0] - 2018-08-08
### Added
- [NEW OP] expand_dims, min, acos, asin, add_n
- Add parse_from_string support. Parse tensorflow pbtext files into tensor_stream

### Fixes
- Tweaks to GradientDescentOptimizer to expose additional methods based on tensorflow

## [0.6.0] - 2018-07-21
### Added
- [NEW OP] fill, floor_div, dynamic_stitch, mod, range, size, squared_difference

### Fixes
- [General] Some auto-differentation fixes
- [softmax_cross_entropy_with_logits_v2] Use numerically stable way of calculating values
- Other fixes related to shape computation

## [0.5.1] - 2018-06-27
### Added
- Added support for control_dependencies
- [NEW OP] floor, ceil

### Fixes
- fixed variable assignment of value sometimes not working
- variable assignment now checks for data types properly

## [0.5.0] - 2018-06-25
### Added
- [OpenCL] boolean types now use short by default
- [OpenCL] Supported multiple OpenCL devices in one model
- Added support for tf.device to alter device placement when using OpenCL
- Internal changes to allow placement of tensor nodes to specific evaluators/devices

### Fixes
- removed dependency on SciRuby distribution for the pure ruby evaluator for better compatibility

## [0.4.1] - 2018-06-17
### Fixes
- [OpenCL] disable program writes to /tmp

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