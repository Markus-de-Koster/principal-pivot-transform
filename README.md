# Principal Pivot Transform
Implementation of Principal Pivot Transforms (PPT), often referred to as exchange or sweep operator.
Check out references section for explanations.

## Install package locally
- Get into your virtual environment
`source path/to/venv/bin/activate`
- setup ppt package
`python3 setup.py bdist_wheel`
- install package from created dist directory
`pip install ./dist/PrincipalPivotTransform-0.1.0-py3-none-any.whl` (adjust Version number / path)
if for some reason the import is not recognized correctly, it might help to force the python interpreter by running:
`python3 -m pip install ./dist/PrincipalPivotTransform-0.1.0-py3-none-any.whl` (adjust Version number / path)
- you should be able to import the project in your code
`import ppt.ppt as ppt`

## Future work
- Complete implementation of generalized PPT
- Improve test case coverage
- Fix import


## References
For this implementation mostly source 2, 3 and 5 are used.

1. W. Wanicharpichat and S. Pompan, “Generalized Pivot Transforms via Lower-Left Hand Corner Submatrix,” May 2016.
2. M. Rajesh Kannan and R. B. Bapat, “Generalized principal pivot transform,” Linear Algebra and its Applications, vol. 454, pp. 49–56, Aug. 2014, doi: 10.1016/j.laa.2014.04.015.
3. K. Kamaraj, P. S. Johnson, and S. M. Naik, “Generalized principal pivot transform and its inheritance properties,” J Anal, vol. 30, no. 3, pp. 1241–1256, Sep. 2022, doi: 10.1007/s41478-022-00399-w.
4. J. E. Pascoe and R. Tully-Doyle, “Monotonicity of the principal pivot transform.” arXiv, Aug. 12, 2021. Accessed: Oct. 11, 2022. [Online]. Available: http://arxiv.org/abs/2108.05910
5. M. J. Tsatsomeros, “Principal pivot transforms: properties and applications,” Linear Algebra and its Applications, vol. 307, no. 1–3, pp. 151–165, Mar. 2000, doi: 10.1016/S0024-3795(99)00281-5.
6. K. Bisht and S. K.C., “Pseudo Principal Pivot Transform: The Group Inverse Case,” May 2016.
7. K. Sivakumar, R. G, and K. Bisht, “Pseudo Schur complements, pseudo principal pivot transforms and their inheritance properties,” The Electronic Journal of Linear Algebra, vol. 30, pp. 455–477, Feb. 2015, doi: 10.13001/1081-3810.2825.


## Contributing
- You are welcome to contribute, especially to fix errors you notice
