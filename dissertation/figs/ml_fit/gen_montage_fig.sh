#!/bin/bash
##
# DESCRIPTION: Generate a montage of images on a structured grid for latex
##

# prefix="./ktau_regression_"
prefix=$1
suffix=".png "
# suffix2="_lab.png "

# gen montage fig list
for i in {1..25};
do
    #convert $i$prefix$suffix   -background Black  -fill White -pointsize 14 label:'Pin '$i \
    #          -gravity Center -append    $i$prefix$suffix2

    image_list+=$prefix$i$suffix
    # image_list+=$i$prefix$suffix2
done

# resize images
w=480  # height and width of each tile in montage
h=720
s=0    # tile border size
montage ${image_list[@]} -geometry $h\x$w+$s+$s montage.png

# reduce size of final image
INFILE=montage.png
OUTFILE=montage_sm.pdf

convert $INFILE $OUTFILE
# convert $INFILE -compress lzw eps2:$OUTFILE
# convert $INFILE -compress lzw -resize 80% eps2:$OUTFILE
