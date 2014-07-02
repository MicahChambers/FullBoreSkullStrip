#!/bin/bash

sub=06

function labeltomask() {
	label=$1
	labelmap=$2
	out=$3
	fslmaths $labelmap -sub $label  -abs -mul -1 -add 1 -thr .5 -bin $out -odt short
}

labeltomask 6 ${sub}_label.nii.gz ${sub}_6.nii.gz
labeltomask 2 ${sub}_label.nii.gz ${sub}_2.nii.gz
labeltomask 11 ${sub}_label.nii.gz ${sub}_11.nii.gz
labeltomask 12 ${sub}_label.nii.gz ${sub}_12.nii.gz

# merge together into brainmask
fslmaths ${sub}_2.nii.gz -add ${sub}_12.nii.gz -add ${sub}_6.nii.gz -add ${sub}_11.nii.gz -bin ${sub}_brainmask.nii.gz -odt short 
