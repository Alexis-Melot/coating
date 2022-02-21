# coating
Study effect of coating

Assuming you have one file G4_MEA.h5 with 5 coating configurations, first you need to create the results files by running spyking-circus :

> spyking-circus G4_MEA.h5
> spyking-circus G4_MEA.h5 -m thresholding

You need to edit the .params first to be sure that [no_edits] filter_done is set to False, if you start with unfiltered data for the first time. 

Once the data have been generated, run the script.py to make the plots
