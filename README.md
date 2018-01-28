# Run-DIC #
**Mikhail Khrenov**
*June 2017 - January 2018*

The final code used for the paper "A Novel Single Camera Robotic Approach for Three-Dimensional 
Digital Image Correlation with Targetless Extrinsic Calibration and Expanded View Angles". Work done 
at the [Multi-Scale Measurements Laboratory](www.robotics.umd.edu/labs/multi-scale-measurements-laboratory)
of the University of Maryland's Clark School of Engineering, Department of Mechanical Engineering.

Mentored by Dr. Hugh A. Bruck.

The python files and functions contained are as follows:

- transforms.py - Tools for calculating relative poses given robotic parameters
    - `transform(points, out)` - Given an input file of *n* points and an output file, calculates *n-1*
     transformations suitable for use in Vic-3D for points 2 through *n* relative to the first point.
    - `convert(x, y, z, pitch, roll, xt, yt, zt, pitch_t, roll_t)` - Given an initial and final pose specification
     (the five parameters used by RoboCIM), returns the appropriate relative pose for use in Vic-3D. 
     Employed by `transform`.
- merge.py - Tool for merging Vic-3D data files
    - `merge_out(output, [source_1, ... source_n])` - Given an output filename and n source files 
     (Vic-3D *.out* files), puts the combined data of *source_1* through *source_n* into *out*.
- expand.py - Tools for automating correlation and combining 3+ images for a single expanded view angle dataset
    - `auto_correlate(directory, img_prefix, [position_1, ... position_n])` - Given a working directory, Vic-Snap 
     image prefix, and *n* positions, automatically performs *n-1* pairwise correlations.
    - `combine(directory, img_prefix, [position_1, ... position_n])` - Given a working directory, Vic-Snap 
     image prefix, and *n* positions, with correlations already having been performed for the same list of positions
     by `auto_correlate`, combines *n-1* data files into a single data file, calculating transformation needed to
     stitch data as it goes.
    - `gen_params(prev_data, table)` - given two sets of data, the second of which's coordinate system is the
     target for the second, and where the two have physically overlapping points, first estimates the relative
     transformation using a six-point approximation, then minimizes the mean squared distance between all overlapping
     points with the previous approximation as the initial guess to return the relative transformation for stitching.
- radius.py - Tool for calculating the radius of cylindrical data
    - `find_radius_error(directory, img_prefix, num)` - Given a working directory, Vic-Snap image prefix, and
     position number, calculates the radius of the cylinder specified by the data point-cloud using least sum of 
     squares optimization over the radius center vector, point, and radius.

**note:** expand.py uses Windows specific automation libraries and shell calls
